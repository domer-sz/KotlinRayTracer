// Path tracer kernel for CUDA, compiled at run time by NVRTC (gpu/CudaContext): a port of
// Camera.rayColor / Sphere.hit / Aabb.hit / the material and texture classes. The recursion of
// rayColor becomes a throughput loop, and every random draw keeps the order and the
// distribution the CPU used so the noise looks the same.
//
// It began as a line-for-line translation of the OpenCL kernel; the helpers below stand in for
// what OpenCL C has built in - vector arithmetic on float3, dot / cross / length / clamp.
// float3 here is 12 bytes (OpenCL's is 16): no buffer is read through a struct holding one,
// only through float, int, int2 and float4 pointers, so the host layouts are unchanged.

#define M_PI_F 3.14159265358979323846f
#define INFINITY __int_as_float(0x7f800000)

__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ inline float3 operator/(float s, float3 a) { return make_float3(s / a.x, s / a.y, s / a.z); }
__device__ inline float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ inline float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
__device__ inline float3& operator-=(float3& a, float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
__device__ inline float3& operator*=(float3& a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
__device__ inline float3& operator*=(float3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
__device__ inline float3& operator/=(float3& a, float s) { a.x /= s; a.y /= s; a.z /= s; return a; }

__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ inline int clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ inline float3 clamp(float3 a, float lo, float hi) {
    return make_float3(clamp(a.x, lo, hi), clamp(a.y, lo, hi), clamp(a.z, lo, hi));
}
__device__ inline float3 xyz(float4 a) { return make_float3(a.x, a.y, a.z); }


#define LEAF_SPHERE   (-1)
#define LEAF_TRIANGLE (-2)
#define LEAF_QUAD     (-3)
#define LEAF_MEDIUM   (-4)
#define TRIANGLE_PARALLEL_EPSILON 1e-8f
#define STACK_SIZE    64
#define REJECTION_CAP 64
#define RR_MIN_DEPTH  5

// Scene features. GpuRenderer sets them with -D so a scene only pays - in code size and, above
// all, in registers - for what it uses; a feature left undefined is compiled in.
#ifndef HAS_SPHERES
#define HAS_SPHERES 1
#endif
#ifndef HAS_TRIANGLES
#define HAS_TRIANGLES 1
#endif
#ifndef HAS_QUADS
#define HAS_QUADS 1
#endif
#ifndef HAS_MEDIA
#define HAS_MEDIA 1
#endif
#ifndef HAS_MOTION
#define HAS_MOTION 1
#endif
#ifndef HAS_LIGHTS
#define HAS_LIGHTS 1
#endif
#ifndef TEX_CHECKER
#define TEX_CHECKER 1
#endif
#ifndef TEX_IMAGE
#define TEX_IMAGE 1
#endif
#ifndef TEX_NOISE
#define TEX_NOISE 1
#endif
#ifndef TEX_UVCHECKER
#define TEX_UVCHECKER 1
#endif
#ifndef MAT_METAL
#define MAT_METAL 1
#endif
#ifndef MAT_DIELECTRIC
#define MAT_DIELECTRIC 1
#endif
#ifndef MAT_EMISSIVE
#define MAT_EMISSIVE 1
#endif
#ifndef MAT_ISOTROPIC
#define MAT_ISOTROPIC 1
#endif

// 0 renders the image. 1 or 2 render a denoiser guide instead, averaged over the pixel like the
// image: the albedo (1) or the normal (2) of whatever the camera ray hits first.
/** Glass surfaces the quick look (AOV_MODE 3) sees through, and the one it stops at. */
#define PREVIEW_LAYERS 5
#ifndef AOV_MODE
#define AOV_MODE 0
#endif

// ---------------------------------------------------------------- random numbers

typedef struct {
    unsigned long long state;
    unsigned long long inc;
} Rng;

__device__ inline unsigned int rng_next(Rng* rng) {
    unsigned long long old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = (unsigned int)(((old >> 18) ^ old) >> 27);
    unsigned int rot = (unsigned int)(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

// One independent stream per (pixel, sample), so the image does not depend on how the
// work is split into batches.
__device__ inline void rng_init(Rng* rng, unsigned long long seed, unsigned long long pixel, unsigned long long sample) {
    rng->state = 0ULL;
    rng->inc = ((pixel * 0x9E3779B97F4A7C15ULL + sample) << 1) | 1ULL;
    rng_next(rng);
    rng->state += seed + 0x853C49E6748FEA9BULL * (sample + 1ULL);
    rng_next(rng);
}

__device__ inline float rng_float(Rng* rng) {
    return (float)(rng_next(rng) >> 8) * 0x1.0p-24f;   // [0, 1)
}

__device__ inline float rng_range(Rng* rng, float lo, float hi) {
    return rng_float(rng) * (hi - lo) + lo;
}

__device__ inline float3 random_in_unit_sphere(Rng* rng) {
    for (int i = 0; i < REJECTION_CAP; i++) {
        float3 p = make_float3(rng_range(rng, -1.0f, 1.0f), rng_range(rng, -1.0f, 1.0f), rng_range(rng, -1.0f, 1.0f));
        if (dot(p, p) < 1.0f) return p;
    }
    return make_float3(0.0f, 0.0f, 1.0f);
}

__device__ inline float3 random_unit_vector(Rng* rng) {
    float3 p = random_in_unit_sphere(rng);
    return p / length(p);
}

// An orthonormal basis around a normal, so a direction drawn in a z-up frame can be turned
// into world space - the kernel's copy of utils/Onb.
__device__ inline void onb_build(float3 normal, float3* u, float3* v, float3* w) {
    *w = normal / length(normal);
    float3 a = fabsf(w->x) > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 crossed = cross(*w, a);
    *v = crossed / length(crossed);
    *u = cross(*w, *v);
}

__device__ inline float3 random_cosine_direction(Rng* rng) {
    float r1 = rng_float(rng);
    float r2 = rng_float(rng);
    float phi = 2.0f * M_PI_F * r1;
    float root = sqrtf(r2);
    return make_float3(cosf(phi) * root, sinf(phi) * root, sqrtf(1.0f - r2));
}

/** A direction into the cone a sphere of [radius] covers from [distanceSquared] away. */
__device__ inline float3 random_to_sphere(Rng* rng, float radius, float distanceSquared) {
    float r1 = rng_float(rng);
    float r2 = rng_float(rng);
    float z = 1.0f + r2 * (sqrtf(1.0f - radius * radius / distanceSquared) - 1.0f);

    float phi = 2.0f * M_PI_F * r1;
    float root = sqrtf(1.0f - z * z);
    return make_float3(cosf(phi) * root, sinf(phi) * root, z);
}

__device__ inline float3 random_on_hemisphere(Rng* rng, float3 normal) {
    float3 p = random_in_unit_sphere(rng);
    float3 onUnitSphere = p / length(p);
    return dot(onUnitSphere, normal) > 0.0f ? onUnitSphere : -onUnitSphere;
}

__device__ inline float2 random_in_unit_disc(Rng* rng) {
    for (int i = 0; i < REJECTION_CAP; i++) {
        float2 p = make_float2(rng_range(rng, -1.0f, 1.0f), rng_range(rng, -1.0f, 1.0f));
        if (dot(p, p) < 1.0f) return p;
    }
    return make_float2(0.0f, 0.0f);
}

__device__ inline bool near_zero(float3 v) {
    const float s = 1e-8f;
    return fabsf(v.x) < s && fabsf(v.y) < s && fabsf(v.z) < s;
}

// ---------------------------------------------------------------- geometry

typedef struct {
    float3 point;
    float3 normal;
    float t;
    float u;
    float v;
    int material;
    bool frontFace;
} HitRecord;

// ---------------------------------------------------------------- scene traversal

// Everything a traversal reads, gathered so the helpers below take one argument, not eleven.
typedef struct {
    const float4* __restrict__ nodes;        // 4 float4 per inner node, as BvhBuilder lays them out
    const int2* __restrict__ leaves;         // first ref, count
    const int2* __restrict__ refs;           // kind, index
    const float* __restrict__ spheres;
    const int* __restrict__ sphereMaterials;
    const float4* __restrict__ triangles;    // v0, edge1, edge2
    const int* __restrict__ triangleMaterials;
    const float* __restrict__ quads;
    const int* __restrict__ quadMaterials;
    const int* __restrict__ mediumInts;
    const float* __restrict__ mediumFloats;
} Scene;

// The closest hit found so far. Only what the search itself needs: normals, points and texture
// coordinates are worked out once, for the winner (finish_hit).
typedef struct {
    float t;
    float u;
    float v;
    int kind;                            // 0 while nothing has been hit
    int index;
} Closest;

__device__ inline bool sphere_t(const Scene* sc, int index, float3 origin, float3 direction, float time,
                     float tMin, float tMax, float* tOut) {
    const float* s = sc->spheres + index * 8;
    float3 center = make_float3(s[0], s[1], s[2]);
    if (HAS_MOTION) center += time * make_float3(s[3], s[4], s[5]);
    float radius = s[6];

    float3 oc = origin - center;
    float a = dot(direction, direction);
    float h = dot(direction, -oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = h * h - a * c;
    if (discriminant < 0.0f) return false;

    float sqrtd = sqrtf(discriminant);
    float root = (h - sqrtd) / a;
    if (!(root > tMin && root < tMax)) {
        root = (h + sqrtd) / a;
        if (!(root > tMin && root < tMax)) return false;
    }
    *tOut = root;
    return true;
}

// Triangle.hit: Moller-Trumbore, with the barycentric coordinates reported as u/v.
__device__ inline bool triangle_t(const Scene* sc, int index, float3 origin, float3 direction,
                       float tMin, float tMax, float* tOut, float* uOut, float* vOut) {
    const float4* t = sc->triangles + index * 3;
    float3 v0 = xyz(t[0]);
    float3 edge1 = xyz(t[1]);
    float3 edge2 = xyz(t[2]);

    float3 pvec = cross(direction, edge2);
    float determinant = dot(edge1, pvec);
    if (fabsf(determinant) < TRIANGLE_PARALLEL_EPSILON) return false;

    float inverseDeterminant = 1.0f / determinant;
    float3 tvec = origin - v0;
    float u = dot(tvec, pvec) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, edge1);
    float v = dot(direction, qvec) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f) return false;

    float root = dot(edge2, qvec) * inverseDeterminant;
    if (!(root > tMin && root < tMax)) return false;
    *tOut = root;
    *uOut = u;
    *vOut = v;
    return true;
}

// Quad.hit: intersect the plane, then express the hit in the quad's own u/v basis.
__device__ inline bool quad_t(const Scene* sc, int index, float3 origin, float3 direction,
                   float tMin, float tMax, float* tOut, float* uOut, float* vOut) {
    const float* data = sc->quads + index * 16;
    float3 Q = make_float3(data[0], data[1], data[2]);
    float3 u = make_float3(data[3], data[4], data[5]);
    float3 v = make_float3(data[6], data[7], data[8]);
    float3 w = make_float3(data[9], data[10], data[11]);
    float3 normal = make_float3(data[12], data[13], data[14]);
    float D = data[15];

    float denominator = dot(normal, direction);
    if (fabsf(denominator) < TRIANGLE_PARALLEL_EPSILON) return false;

    float t = (D - dot(normal, origin)) / denominator;
    if (!(t >= tMin && t <= tMax)) return false;

    float3 planarHit = origin + t * direction - Q;
    float alpha = dot(w, cross(planarHit, v));
    float beta = dot(w, cross(u, planarHit));
    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return false;
    *tOut = t;
    *uOut = alpha;
    *vOut = beta;
    return true;
}

// Tests the plain primitives of one leaf, narrowing best->t as it goes.
__device__ inline void leaf_plain(const Scene* sc, int code, float3 origin, float3 direction, float time,
                       float tMin, Closest* best) {
    int2 leaf = sc->leaves[~code];
    for (int i = leaf.x; i < leaf.x + leaf.y; i++) {
        int2 ref = sc->refs[i];
        float t, u = 0.0f, v = 0.0f;
        bool hit;
        if (HAS_TRIANGLES && ref.x == LEAF_TRIANGLE) hit = triangle_t(sc, ref.y, origin, direction, tMin, best->t, &t, &u, &v);
        else if (HAS_SPHERES && ref.x == LEAF_SPHERE) hit = sphere_t(sc, ref.y, origin, direction, time, tMin, best->t, &t);
        else if (HAS_QUADS && ref.x == LEAF_QUAD) hit = quad_t(sc, ref.y, origin, direction, tMin, best->t, &t, &u, &v);
        else hit = false;
        if (hit) {
            best->t = t; best->u = u; best->v = v; best->kind = ref.x; best->index = ref.y;
        }
    }
}

/*
 * Closest-hit traversal. Each node holds its two children's boxes, so one 64-byte read tests
 * both; leaves are opened on the spot, and of two inner children the nearer is taken first while
 * the other waits on the stack. LEAF_FN is the leaf test - plain geometry, or geometry and fog.
 *
 * The slab test multiplies by a precomputed 1/direction. A ray parallel to a slab plane gives
 * 0 * inf = NaN when it starts exactly on the plane; fmin/fmax return their other operand for a
 * NaN, so such an axis simply stops constraining the interval instead of poisoning it.
 */
#define BVH_TRAVERSE(LEAF_FN)                                                                  \
    {                                                                                          \
        float3 inv = 1.0f / direction;                                                         \
        int stack[STACK_SIZE];                                                                 \
        int sp = 0;                                                                            \
        int node = root;                                                                       \
        while (true) {                                                                         \
            const float4* n = sc->nodes + node * 4;                                   \
            float4 n0 = n[0];                                                                  \
            float4 n1 = n[1];                                                                  \
            float4 n2 = n[2];                                                                  \
            float4 n3 = n[3]; int2 codes = make_int2(__float_as_int(n3.x), __float_as_int(n3.y));                                                     \
            float aLoX = (n0.x - origin.x) * inv.x, aHiX = (n0.y - origin.x) * inv.x;          \
            float aLoY = (n0.z - origin.y) * inv.y, aHiY = (n0.w - origin.y) * inv.y;          \
            float aLoZ = (n2.x - origin.z) * inv.z, aHiZ = (n2.y - origin.z) * inv.z;          \
            float bLoX = (n1.x - origin.x) * inv.x, bHiX = (n1.y - origin.x) * inv.x;          \
            float bLoY = (n1.z - origin.y) * inv.y, bHiY = (n1.w - origin.y) * inv.y;          \
            float bLoZ = (n2.z - origin.z) * inv.z, bHiZ = (n2.w - origin.z) * inv.z;          \
            float aNear = fmaxf(fmaxf(fminf(aLoX, aHiX), fminf(aLoY, aHiY)), fmaxf(fminf(aLoZ, aHiZ), tMin)); \
            float aFar = fminf(fminf(fmaxf(aLoX, aHiX), fmaxf(aLoY, aHiY)), fminf(fmaxf(aLoZ, aHiZ), best->t)); \
            float bNear = fmaxf(fmaxf(fminf(bLoX, bHiX), fminf(bLoY, bHiY)), fmaxf(fminf(bLoZ, bHiZ), tMin)); \
            float bFar = fminf(fminf(fmaxf(bLoX, bHiX), fmaxf(bLoY, bHiY)), fminf(fmaxf(bLoZ, bHiZ), best->t)); \
            bool hitA = aFar >= aNear;                                                         \
            bool hitB = bFar >= bNear;                                                         \
            if (hitA && codes.x < 0) { LEAF_FN(codes.x); hitA = false; }                        \
            if (hitB && bNear > best->t) hitB = false;                                         \
            if (hitB && codes.y < 0) { LEAF_FN(codes.y); hitB = false; }                        \
            if (hitA && aNear > best->t) hitA = false;                                         \
            if (hitA && hitB) {                                                                \
                bool bFirst = bNear < aNear;                                                   \
                stack[sp++] = bFirst ? codes.x : codes.y;                                      \
                node = bFirst ? codes.y : codes.x;                                             \
            } else if (hitA) {                                                                 \
                node = codes.x;                                                                \
            } else if (hitB) {                                                                 \
                node = codes.y;                                                                \
            } else {                                                                           \
                if (sp == 0) break;                                                            \
                node = stack[--sp];                                                            \
            }                                                                                  \
        }                                                                                      \
    }

#define PLAIN_LEAF(code) leaf_plain(sc, code, origin, direction, time, tMin, best)

// Nearest t along the ray over a tree of plain geometry - a volume's boundary. INFINITY on a miss.
__device__ inline float boundary_t(const Scene* sc, int root, float3 origin, float3 direction, float time,
                        float tMin, float tMax) {
    Closest found;
    found.t = tMax;
    found.kind = 0;
    Closest* best = &found;
    BVH_TRAVERSE(PLAIN_LEAF)
    return found.kind != 0 ? found.t : INFINITY;
}

// ConstantMedium.hit: find where the ray enters and leaves the boundary, then stop it at a
// random depth drawn from the density. A depth past the far wall means the ray passes through.
__device__ inline bool medium_t(const Scene* sc, int index, float3 origin, float3 direction, float time,
                     float tMin, float tMax, Rng* rng, float* tOut) {
    int boundaryRoot = sc->mediumInts[index * 2];
    float negativeInverseDensity = sc->mediumFloats[index];

    float entry = boundary_t(sc, boundaryRoot, origin, direction, time, -INFINITY, INFINITY);
    if (entry == INFINITY) return false;
    float exitT = boundary_t(sc, boundaryRoot, origin, direction, time, entry + 0.0001f, INFINITY);
    if (exitT == INFINITY) return false;

    float start = entry < tMin ? tMin : entry;
    float end = exitT > tMax ? tMax : exitT;
    if (start >= end) return false;
    if (start < 0.0f) start = 0.0f;

    float rayLength = length(direction);
    float distanceInside = (end - start) * rayLength;
    float hitDistance = negativeInverseDensity * logf(rng_float(rng));
    if (hitDistance > distanceInside) return false;

    *tOut = start + hitDistance / rayLength;
    return true;
}

// A leaf of the scene tree: plain geometry, plus the volumes that only the scene tree holds.
__device__ inline void leaf_world(const Scene* sc, int code, float3 origin, float3 direction, float time,
                       float tMin, Rng* rng, Closest* best) {
    int2 leaf = sc->leaves[~code];
    for (int i = leaf.x; i < leaf.x + leaf.y; i++) {
        int2 ref = sc->refs[i];
        float t, u = 0.0f, v = 0.0f;
        bool hit;
        if (HAS_TRIANGLES && ref.x == LEAF_TRIANGLE) hit = triangle_t(sc, ref.y, origin, direction, tMin, best->t, &t, &u, &v);
        else if (HAS_SPHERES && ref.x == LEAF_SPHERE) hit = sphere_t(sc, ref.y, origin, direction, time, tMin, best->t, &t);
        else if (HAS_QUADS && ref.x == LEAF_QUAD) hit = quad_t(sc, ref.y, origin, direction, tMin, best->t, &t, &u, &v);
        else if (HAS_MEDIA) hit = medium_t(sc, ref.y, origin, direction, time, tMin, best->t, rng, &t);
        else hit = false;
        if (hit) {
            best->t = t; best->u = u; best->v = v; best->kind = ref.x; best->index = ref.y;
        }
    }
}

#define WORLD_LEAF(code) leaf_world(sc, code, origin, direction, time, tMin, rng, best)

// Sphere.getSphereUv, from the outward normal.
__device__ inline void sphere_uv(float3 outwardNormal, HitRecord* rec) {
    float theta = acosf(-outwardNormal.y);
    float phi = atan2f(-outwardNormal.z, outwardNormal.x) + M_PI_F;
    rec->u = phi / (2.0f * M_PI_F);
    rec->v = theta / M_PI_F;
}

// Everything shading needs about the winning hit, computed once.
__device__ inline void finish_hit(const Scene* sc, const Closest* best, float3 origin, float3 direction,
                       float time, HitRecord* rec) {
    rec->t = best->t;
    rec->point = origin + best->t * direction;
    rec->u = best->u;
    rec->v = best->v;
    float3 outwardNormal;

    if (HAS_SPHERES && best->kind == LEAF_SPHERE) {
        const float* s = sc->spheres + best->index * 8;
        float3 center = make_float3(s[0], s[1], s[2]);
    if (HAS_MOTION) center += time * make_float3(s[3], s[4], s[5]);
        outwardNormal = (rec->point - center) / s[6];
        sphere_uv(outwardNormal, rec);
        rec->material = sc->sphereMaterials[best->index];
    } else if (HAS_TRIANGLES && best->kind == LEAF_TRIANGLE) {
        const float4* t = sc->triangles + best->index * 3;
        outwardNormal = cross(xyz(t[1]), xyz(t[2]));
        outwardNormal = outwardNormal / length(outwardNormal);
        rec->material = sc->triangleMaterials[best->index];
    } else if (HAS_QUADS && best->kind == LEAF_QUAD) {
        const float* data = sc->quads + best->index * 16;
        outwardNormal = make_float3(data[12], data[13], data[14]);
        rec->material = sc->quadMaterials[best->index];
    } else {
        // Fog has no surface: the normal is arbitrary and the ray always counts as outside.
        rec->normal = make_float3(1.0f, 0.0f, 0.0f);
        rec->frontFace = true;
        rec->material = sc->mediumInts[best->index * 2 + 1];
        rec->u = 0.0f;
        rec->v = 0.0f;
        return;
    }
    rec->frontFace = dot(direction, outwardNormal) < 0.0f;
    rec->normal = rec->frontFace ? outwardNormal : -outwardNormal;
}

__device__ inline bool world_hit(const Scene* sc, int root, float3 origin, float3 direction, float time,
                      float tMin, float tMax, Rng* rng, HitRecord* rec) {
    Closest found;
    found.t = tMax;
    found.kind = 0;
    Closest* best = &found;
    BVH_TRAVERSE(WORLD_LEAF)
    if (found.kind == 0) return false;
    finish_hit(sc, &found, origin, direction, time, rec);
    return true;
}

// ---------------------------------------------------------------- noise

#define PERLIN_BLOCK 768
#define TURBULENCE_DEPTH 7

__device__ inline float perlin_noise(const float* vectors, const int* permutations,
                          int block, float3 p) {
    float u = p.x - floorf(p.x);
    float v = p.y - floorf(p.y);
    float w = p.z - floorf(p.z);

    int i = (int)floorf(p.x);
    int j = (int)floorf(p.y);
    int k = (int)floorf(p.z);

    // Hermite smoothing, so the lattice does not show through as blocks.
    float uu = u * u * (3.0f - 2.0f * u);
    float vv = v * v * (3.0f - 2.0f * v);
    float ww = w * w * (3.0f - 2.0f * w);

    int permBase = block * PERLIN_BLOCK;
    int vectorBase = block * PERLIN_BLOCK;
    float accumulated = 0.0f;

    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for (int dk = 0; dk < 2; dk++) {
                int index = permutations[permBase + ((i + di) & 255)]
                          ^ permutations[permBase + 256 + ((j + dj) & 255)]
                          ^ permutations[permBase + 512 + ((k + dk) & 255)];
                float3 corner = make_float3(vectors[vectorBase + index * 3],
                                         vectors[vectorBase + index * 3 + 1],
                                         vectors[vectorBase + index * 3 + 2]);
                float3 weight = make_float3(u - di, v - dj, w - dk);
                accumulated += (di * uu + (1 - di) * (1.0f - uu))
                             * (dj * vv + (1 - dj) * (1.0f - vv))
                             * (dk * ww + (1 - dk) * (1.0f - ww))
                             * dot(corner, weight);
            }
        }
    }
    return accumulated;
}

__device__ inline float perlin_turbulence(const float* vectors, const int* permutations,
                               int block, float3 p) {
    float accumulated = 0.0f;
    float3 sample = p;
    float weight = 1.0f;
    for (int octave = 0; octave < TURBULENCE_DEPTH; octave++) {
        accumulated += weight * perlin_noise(vectors, permutations, block, sample);
        weight *= 0.5f;
        sample *= 2.0f;
    }
    return fabsf(accumulated);
}

// ---------------------------------------------------------------- textures

__device__ inline float3 texture_value(const int* texI, const float* texF,
                            const unsigned char* images,
                            const float* perlinVectors, const int* perlinPermutations,
                            int index, float u, float v, float3 p) {
    for (int hop = 0; hop < 8; hop++) {
        int type = texI[index * 4];

        if (type == 0) {                                  // SolidColorTexture
            const float* f = texF + index * 4;
            return make_float3(f[0], f[1], f[2]);
        }
        if (TEX_CHECKER && type == 1) {                                  // CheckerTexture
            float inverted = texF[index * 4];
            int xInteger = (int)floorf(inverted * p.x);
            int yInteger = (int)floorf(inverted * p.y);
            int zInteger = (int)floorf(inverted * p.z);
            bool isEven = ((xInteger + yInteger + zInteger) % 2) == 0;
            index = isEven ? texI[index * 4 + 1] : texI[index * 4 + 2];
            continue;
        }
        if (TEX_IMAGE && type == 2) {                                  // ImageTexture
            int offset = texI[index * 4 + 1];
            int width = texI[index * 4 + 2];
            int height = texI[index * 4 + 3];

            float uu = clamp(u, 0.0f, 1.0f);
            float vv = 1.0f - clamp(v, 0.0f, 1.0f);       // flip V to image coordinates

            int i = (int)(uu * (float)width);
            int j = (int)(vv * (float)height);
            i = i < 0 ? 0 : (i < width ? i : width - 1);
            j = j < 0 ? 0 : (j < height ? j : height - 1);

            int at = offset + (j * width + i) * 3;
            return make_float3((float)images[at], (float)images[at + 1], (float)images[at + 2]) * (1.0f / 255.0f);
        }
        if (TEX_UVCHECKER && type == 5) {                              // UvCheckerTexture
            float cellsU = texF[index * 4];
            float cellsV = texF[index * 4 + 1];
            int cellU = clamp((int)floorf(u * cellsU), 0, (int)cellsU - 1);
            int cellV = clamp((int)floorf(v * cellsV), 0, (int)cellsV - 1);
            index = ((cellU + cellV) % 2 == 0) ? texI[index * 4 + 1] : texI[index * 4 + 2];
            continue;
        }
        if (TEX_NOISE && type == 4) {                                  // NoiseTexture
            float scale = texF[index * 4];
            int block = texI[index * 4 + 1];
            float value = 0.5f * (1.0f + sinf(scale * p.z
                + 10.0f * perlin_turbulence(perlinVectors, perlinPermutations, block, p)));
            return make_float3(value, value, value);
        }
        break;                                            // image that failed to load
    }
    return make_float3(0.0f, 1.0f, 1.0f);
}

// ---------------------------------------------------------------- materials

__device__ inline float reflectance(float cosine, float ri) {
    float r0 = (1.0f - ri) / (1.0f + ri);
    r0 = r0 * r0;
    float t = 1.0f - cosine;
    float t2 = t * t;
    return r0 + (1.0f - r0) * (t2 * t2 * t);
}

__device__ inline float3 refract_dir(float3 uv, float3 n, float etaiOverEtat) {
    float cosTheta = fminf(dot(-uv, n), 1.0f);
    float3 rOutPerp = (n * cosTheta + uv) * etaiOverEtat;
    float3 rOutParallel = -(n * sqrtf(fabsf(1.0f - dot(rOutPerp, rOutPerp))));
    return rOutPerp + rOutParallel;
}

/** Light a surface gives off by itself; only DiffuseLight has any. */
__device__ inline float3 material_emitted(const int* matI,
                               const int* texI, const float* texF,
                               const unsigned char* images,
                               const float* perlinVectors, const int* perlinPermutations,
                               const HitRecord* rec) {
    // Only emissive materials, and only through their front: a lamp does not light the room
    // through the back of the ceiling.
    if (!MAT_EMISSIVE || matI[rec->material * 2] != 3 || !rec->frontFace) return make_float3(0.0f, 0.0f, 0.0f);
    return texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                         matI[rec->material * 2 + 1], rec->u, rec->v, rec->point);
}

/**
 * What a material does with a ray, the kernel's copy of materials/ScatterRecord: either a
 * direction drawn from a density ([pdfKind], to be divided out), or the single direction a
 * mirror or a piece of glass sends the ray into ([skipPdf]).
 */
#define PDF_COSINE 0
#define PDF_SPHERE 1

typedef struct {
    float3 attenuation;
    float3 direction;
    float3 pdfAxis;                                       // the cosine density's normal
    bool skipPdf;
    int pdfKind;
} ScatterRec;

__device__ inline float scatter_pdf_value(const ScatterRec* srec, float3 direction) {
    if (srec->pdfKind == PDF_SPHERE) return 1.0f / (4.0f * M_PI_F);
    float cosine = dot(direction / length(direction), srec->pdfAxis);
    return fmaxf(0.0f, cosine / M_PI_F);
}

/** The surface's own scattering density for a direction it was handed. */
__device__ inline float scattering_pdf(int materialType, float3 normal, float3 scattered) {
    if (materialType == 4) return 1.0f / (4.0f * M_PI_F);          // Isotropic
    float cosine = dot(normal, scattered / length(scattered));
    return cosine < 0.0f ? 0.0f : cosine / M_PI_F;                 // Lambertian
}

__device__ inline bool scatter(const int* matI, const float* matF,
                    const int* texI, const float* texF,
                    const unsigned char* images,
                    const float* perlinVectors, const int* perlinPermutations,
                    const HitRecord* rec, float3 inDirection, Rng* rng,
                    ScatterRec* srec) {
    int type = matI[rec->material * 2];
    int texture = matI[rec->material * 2 + 1];

    srec->skipPdf = false;
    srec->pdfKind = PDF_COSINE;
    srec->pdfAxis = rec->normal;

    if (MAT_EMISSIVE && type == 3) return false;                          // DiffuseLight: the path ends here

    if (MAT_ISOTROPIC && type == 4) {                                      // Isotropic: fog scatters anywhere
        srec->attenuation = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                          texture, rec->u, rec->v, rec->point);
        srec->direction = random_unit_vector(rng);
        srec->pdfKind = PDF_SPHERE;
        return true;
    }

    if (type == 0) {                                      // Lambertian, sampled by cosine
        float3 u, v, w;
        onb_build(rec->normal, &u, &v, &w);
        float3 sampled = random_cosine_direction(rng);   // "local" is a keyword in OpenCL C
        srec->attenuation = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                          texture, rec->u, rec->v, rec->point);
        srec->direction = u * sampled.x + v * sampled.y + w * sampled.z;
        return true;
    }

    float3 unitDirection = inDirection / length(inDirection);
    srec->skipPdf = true;                                 // specular from here on

    if (MAT_METAL && type == 1) {                                      // Metal
        float fuzz = matF[rec->material * 2];
        float3 reflected = unitDirection - rec->normal * (2.0f * dot(unitDirection, rec->normal));
        reflected = reflected / length(reflected);
        srec->attenuation = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                          texture, rec->u, rec->v, rec->point);
        srec->direction = reflected + random_unit_vector(rng) * fuzz;
        return true;
    }

    // Dielectric - the only material left, unless the scene has none.
    if (!MAT_DIELECTRIC) return false;
    float ri = matF[rec->material * 2 + 1];
    float refractionRatio = rec->frontFace ? (1.0f / ri) : ri;
    float cosTheta = fminf(dot(-unitDirection, rec->normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    bool cannotRefract = refractionRatio * sinTheta > 1.0f;
    // Short-circuit: the random draw only happens when refraction is possible, as on the CPU.
    bool useReflect = cannotRefract || (reflectance(cosTheta, refractionRatio) > rng_float(rng));

    srec->attenuation = make_float3(1.0f, 1.0f, 1.0f);
    srec->direction = useReflect
        ? unitDirection - rec->normal * (2.0f * dot(unitDirection, rec->normal))
        : refract_dir(unitDirection, rec->normal, refractionRatio);
    return true;
}

// The surface colour a denoiser should keep: the texture's own colour, white for glass (it has
// none of its own), and lamps clamped to white.
__device__ inline float3 aov_albedo(const int* matI,
                         const int* texI, const float* texF,
                         const unsigned char* images,
                         const float* perlinVectors, const int* perlinPermutations,
                         const HitRecord* rec) {
    int type = matI[rec->material * 2];
    if (type == 2) return make_float3(1.0f, 1.0f, 1.0f);
    float3 colour = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                  matI[rec->material * 2 + 1], rec->u, rec->v, rec->point);
    return clamp(colour, 0.0f, 1.0f);
}

/**
 * The quick look for setting a scene up (AOV_MODE 3): the first surface's own colour under a
 * light at the camera, plus a little from above so floors and tops stand apart from walls.
 * No bounces, so no noise to wait out. Lights show flat in their colour, glass pale blue.
 */
__device__ inline float3 preview_shade(const int* matI,
                            const int* texI, const float* texF,
                            const unsigned char* images,
                            const float* perlinVectors, const int* perlinPermutations,
                            const HitRecord* rec, float3 direction) {
    int type = matI[rec->material * 2];
    float3 colour = aov_albedo(matI, texI, texF, images, perlinVectors, perlinPermutations, rec);
    if (type == 3) return colour;
    if (type == 2) colour = make_float3(0.70f, 0.82f, 0.92f);
    float facing = fabsf(dot(rec->normal, direction)) / length(direction);
    return colour * (0.18f + 0.62f * facing + 0.2f * fmaxf(rec->normal.y, 0.0f));
}

// ---------------------------------------------------------------- sampling the lights

/** A direction from [origin] towards a random point on one of the lights, picked evenly. */
__device__ inline float3 light_random(const float* quads, const float* spheres,
                           const int* lights, int lightCount, float3 origin, Rng* rng) {
    int pick = (int)(rng_float(rng) * (float)lightCount);
    if (pick >= lightCount) pick = lightCount - 1;

    int kind = lights[pick * 2];
    int index = lights[pick * 2 + 1];

    if (kind == LEAF_QUAD) {
        const float* data = quads + index * 16;
        float3 Q = make_float3(data[0], data[1], data[2]);
        float3 u = make_float3(data[3], data[4], data[5]);
        float3 v = make_float3(data[6], data[7], data[8]);
        float3 point = Q + u * rng_float(rng) + v * rng_float(rng);
        return point - origin;
    }

    const float* data = spheres + index * 8;
    float3 centre = make_float3(data[0], data[1], data[2]);   // lights do not move
    float radius = data[6];
    float3 toCentre = centre - origin;
    float distanceSquared = dot(toCentre, toCentre);
    // From inside - an object standing in a glass sphere - the sphere is all around.
    if (distanceSquared <= radius * radius) return random_unit_vector(rng);

    float3 u, v, w;
    onb_build(toCentre, &u, &v, &w);
    float3 sampled = random_to_sphere(rng, radius, distanceSquared);
    return u * sampled.x + v * sampled.y + w * sampled.z;
}

/** How likely that direction was, averaged over the lights - Hittable.pdfValue in the kernel. */
__device__ inline float light_pdf_value(const Scene* sc, const int* lights, int lightCount,
                             float3 origin, float3 direction) {
    float weight = 1.0f / (float)lightCount;
    float sum = 0.0f;

    for (int i = 0; i < lightCount; i++) {
        int kind = lights[i * 2];
        int index = lights[i * 2 + 1];
        float t, u, v;

        if (kind == LEAF_QUAD) {
            if (!quad_t(sc, index, origin, direction, 0.001f, INFINITY, &t, &u, &v)) continue;
            const float* data = sc->quads + index * 16;
            float3 normal = make_float3(data[12], data[13], data[14]);
            float3 edgeU = make_float3(data[3], data[4], data[5]);
            float3 edgeV = make_float3(data[6], data[7], data[8]);
            float area = length(cross(edgeU, edgeV));

            float distanceSquared = t * t * dot(direction, direction);
            // Which side the ray arrives from does not matter here: the cosine is taken absolute.
            float cosine = fabsf(dot(direction, normal) / length(direction));
            if (cosine < TRIANGLE_PARALLEL_EPSILON) continue;
            sum += weight * distanceSquared / (cosine * area);
        } else {
            if (!sphere_t(sc, index, origin, direction, 0.0f, 0.001f, INFINITY, &t)) continue;
            const float* data = sc->spheres + index * 8;
            float3 centre = make_float3(data[0], data[1], data[2]);
            float radius = data[6];
            float3 toCentre = centre - origin;
            float distanceSquared = dot(toCentre, toCentre);
            // Seen from inside it covers every direction; outside, the cone it fills.
            float solidAngle = 4.0f * M_PI_F;
            if (distanceSquared > radius * radius) {
                float cosThetaMax = sqrtf(1.0f - radius * radius / distanceSquared);
                solidAngle = 2.0f * M_PI_F * (1.0f - cosThetaMax);
            }
            sum += weight / solidAngle;
        }
    }
    return sum;
}

// ---------------------------------------------------------------- entry point

extern "C" __global__ void render(const float* __restrict__ cam,
                     const float4* __restrict__ nodes,
                     const int2* __restrict__ leaves,
                     const int2* __restrict__ refs,
                     const float* __restrict__ spheres,
                     const int* __restrict__ sphereMaterials,
                     const float4* __restrict__ triangles,
                     const int* __restrict__ triangleMaterials,
                     const float* __restrict__ quads,
                     const int* __restrict__ quadMaterials,
                     const int* __restrict__ mediumInts,
                     const float* __restrict__ mediumFloats,
                     const int* __restrict__ lights,
                     const int* __restrict__ matI,
                     const float* __restrict__ matF,
                     const int* __restrict__ texI,
                     const float* __restrict__ texF,
                     const unsigned char* __restrict__ images,
                     const float* __restrict__ perlinVectors,
                     const int* __restrict__ perlinPermutations,
                     float* out,
                     const int width,
                     const int height,
                     const int sampleStart,
                     const int sampleCount,
                     const int maxReflectionDepth,
                     const float pixelSamplesScale,
                     const int rootNode,
                     const int lightCount,
                     const unsigned long long seed) {
    // Work items come in warps of 32, each covering an 8x4 tile of pixels.
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int tile = gid >> 5;
    int lane = gid & 31;
    int tilesX = (width + 7) >> 3;
    int x = (tile % tilesX) * 8 + (lane & 7);
    int y = (tile / tilesX) * 4 + (lane >> 3);
    if (x >= width || y >= height) return;

    Scene scene;
    scene.nodes = nodes;
    scene.leaves = leaves;
    scene.refs = refs;
    scene.spheres = spheres;
    scene.sphereMaterials = sphereMaterials;
    scene.triangles = triangles;
    scene.triangleMaterials = triangleMaterials;
    scene.quads = quads;
    scene.quadMaterials = quadMaterials;
    scene.mediumInts = mediumInts;
    scene.mediumFloats = mediumFloats;

    float3 cameraCenter = make_float3(cam[0], cam[1], cam[2]);
    float3 pixel00      = make_float3(cam[3], cam[4], cam[5]);
    float3 pixelDeltaU  = make_float3(cam[6], cam[7], cam[8]);
    float3 pixelDeltaV  = make_float3(cam[9], cam[10], cam[11]);
    float3 defocusDiscU = make_float3(cam[12], cam[13], cam[14]);
    float3 defocusDiscV = make_float3(cam[15], cam[16], cam[17]);
    float defocusAngle  = cam[18];
    bool flatBackground = cam[19] > 0.5f;
    float3 background   = make_float3(cam[20], cam[21], cam[22]);

    float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);

    // One bounce per iteration rather than a loop over samples around a loop over bounces: a
    // thread whose path has ended starts its next sample straight away, so the threads of a
    // warp keep running the same bounce code instead of some idling while others start anew.
    // Each sample still draws the same random numbers in the same order.
    int sample = sampleStart;
    // With no bounces allowed every path is black, as on the CPU; the loop below would
    // otherwise trace one bounce before checking.
    int sampleEnd = maxReflectionDepth > 0 ? sampleStart + sampleCount : sample;
    bool fresh = true;
    Rng rng;
    float3 origin, direction, throughput, sampleColor;
    float time;
    int depth;

    while (sample < sampleEnd) {
        if (fresh) {
            rng_init(&rng, seed, (unsigned long long)(y * width + x), (unsigned long long)sample);

            // Camera.getRay
            float offsetX = rng_float(&rng) - 0.5f;
            float offsetY = rng_float(&rng) - 0.5f;
            float3 pixelSample = pixel00
                + pixelDeltaU * ((float)x + offsetX)
                + pixelDeltaV * ((float)y + offsetY);

            origin = cameraCenter;
            if (defocusAngle > 0.0f) {
                float2 p = random_in_unit_disc(&rng);
                origin = cameraCenter + defocusDiscU * p.x + defocusDiscV * p.y;
            }
            direction = pixelSample - origin;
            time = rng_float(&rng);

            // Camera.rayColor as a throughput loop. A path that never escapes the scene
            // within maxReflectionDepth bounces contributes nothing more, as on the CPU.
            throughput = make_float3(1.0f, 1.0f, 1.0f);
            sampleColor = make_float3(0.0f, 0.0f, 0.0f);
            depth = 0;
            fresh = false;
        }

        if (AOV_MODE == 3) {
            // The quick look. Glass is not refracted but seen through, tinted a little at each
            // surface, so what stands behind it can still be placed.
            float3 shaded = make_float3(0.0f, 0.0f, 0.0f);
            float3 carried = make_float3(1.0f, 1.0f, 1.0f);
            for (int layer = 0; layer < PREVIEW_LAYERS; layer++) {
                HitRecord hit;
                if (!world_hit(&scene, rootNode, origin, direction, time, 0.001f, INFINITY, &rng, &hit)) {
                    float3 sky = background;
                    if (!flatBackground) {
                        float alpha = 0.5f * (direction.y / length(direction) + 1.0f);
                        sky = make_float3(1.0f, 1.0f, 1.0f) * (1.0f - alpha) + make_float3(0.5f, 0.7f, 1.0f) * alpha;
                    }
                    shaded += carried * sky;
                    break;
                }
                float3 surface = preview_shade(matI, texI, texF, images, perlinVectors, perlinPermutations, &hit, direction);
                if (matI[hit.material * 2] != 2 || layer == PREVIEW_LAYERS - 1) {
                    shaded += carried * surface;
                    break;
                }
                shaded += carried * 0.25f * surface;
                carried *= 0.75f;
                origin = hit.point;
            }
            pixelColor += shaded;
            sample++;
            fresh = true;
            continue;
        }

        if (AOV_MODE) {
            HitRecord first;
            float3 guide = make_float3(0.0f, 0.0f, 0.0f);
            if (world_hit(&scene, rootNode, origin, direction, time, 0.001f, INFINITY, &rng, &first)) {
                guide = AOV_MODE == 2 ? first.normal
                    : aov_albedo(matI, texI, texF, images, perlinVectors, perlinPermutations, &first);
            }
            pixelColor += guide;
            sample++;
            fresh = true;
            continue;
        }

        bool done = false;
        HitRecord rec;
        if (world_hit(&scene, rootNode, origin, direction, time, 0.001f, INFINITY, &rng, &rec)) {
            sampleColor += throughput * material_emitted(matI, texI, texF, images,
                                                         perlinVectors, perlinPermutations, &rec);
            ScatterRec srec;
            if (!scatter(matI, matF, texI, texF, images, perlinVectors, perlinPermutations,
                         &rec, direction, &rng, &srec)) {
                done = true;
            } else if (srec.skipPdf) {
                throughput *= srec.attenuation;
                origin = rec.point;
                direction = srec.direction;
            } else {
                // Half the samples aimed at a light, half drawn by the surface itself.
                float3 chosen = srec.direction;
                if (HAS_LIGHTS && lightCount > 0 && rng_float(&rng) < 0.5f) {
                    chosen = light_random(quads, spheres, lights, lightCount, rec.point, &rng);
                }

                float density = scatter_pdf_value(&srec, chosen);
                if (HAS_LIGHTS && lightCount > 0) {
                    density = 0.5f * density + 0.5f * light_pdf_value(&scene, lights, lightCount, rec.point, chosen);
                }
                if (density < 1e-8f) {
                    done = true;
                } else {
                    // Weigh the sample by how much the surface actually scatters that way;
                    // that division is what keeps the estimate unbiased.
                    float surface = scattering_pdf(matI[rec.material * 2], rec.normal, chosen);
                    throughput *= srec.attenuation * (surface / density);

                    origin = rec.point;
                    direction = chosen;
                }
            }

            // Russian roulette: past a few bounces, end dim paths at random and boost the
            // survivors by the same odds, so the estimate stays unbiased. A warp waits for
            // its longest path, so cutting the long tail is what keeps the GPU busy.
            if (!done && depth >= RR_MIN_DEPTH) {
                float survival = clamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.25f, 0.95f);
                if (rng_float(&rng) >= survival) done = true;
                else throughput /= survival;
            }
        } else {
            float3 escaped = background;
            if (!flatBackground) {
                float3 unitDirection = direction / length(direction);
                float alpha = 0.5f * (unitDirection.y + 1.0f);
                escaped = make_float3(1.0f, 1.0f, 1.0f) * (1.0f - alpha)
                        + make_float3(0.5f, 0.7f, 1.0f) * alpha;
            }
            sampleColor += throughput * escaped;
            done = true;
        }

        depth++;
        if (done || depth >= maxReflectionDepth) {
            pixelColor += sampleColor;
            sample++;
            fresh = true;
        }
    }

    pixelColor *= pixelSamplesScale;

    // The first slice of samples writes the pixel, later slices add to it.
    int base = (y * width + x) * 3;
    if (sampleStart > 0) pixelColor += make_float3(out[base], out[base + 1], out[base + 2]);
    out[base]     = pixelColor.x;
    out[base + 1] = pixelColor.y;
    out[base + 2] = pixelColor.z;
}
