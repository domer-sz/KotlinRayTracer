// Path tracer kernel for CUDA, compiled at run time by NVRTC (gpu/CudaContext): a port of
// Camera.rayColor / Sphere.hit / Aabb.hit / the material and texture classes. The recursion of
// rayColor becomes a throughput loop, and every random draw keeps the order and the
// distribution the CPU used so the noise looks the same.
//
// It began as a line-for-line translation of the OpenCL kernel; the helpers below stand in for
// what OpenCL C has built in - vector arithmetic on float3, dot / cross / length / clamp.
// float3 here is 12 bytes (OpenCL's is 16): no buffer is read through a struct holding one,
// only through float, int, int2 and float4 pointers, so the host layouts are unchanged.
//
// With -DRT_OPTIX=1 the same path tracer is built as an OptiX program instead (gpu/optix): OptiX
// finds the hits - on an RTX card its RT cores do - and everything else is the code below,
// shared. Only world_hit and the entry point differ.

#ifndef RT_OPTIX
#define RT_OPTIX 0
#endif

#if RT_OPTIX
// NVRTC does not say which host it builds for, and OptiX sizes its device pointers by it.
#if !defined(_WIN64) && !defined(__LP64__)
#define _WIN64 1
#endif
#include <optix.h>
#endif

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
#define LEAF_MESH     (-5)                   // a triangle of a placed mesh, which only OptiX keeps as an instance
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
#ifndef HAS_SMOOTH
#define HAS_SMOOTH 1
#endif
#ifndef HAS_UV
#define HAS_UV 1
#endif
#ifndef HAS_NORMALMAP
#define HAS_NORMALMAP 1
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
#ifndef GUIDES
#define GUIDES 0
#endif
#ifndef AOV_MODE
#define AOV_MODE 0
#endif
// With GUIDES, the albedo guide (AOV_MODE 1) writes the normal guide beside it from the same rays.
#define GUIDES_ON (GUIDES && AOV_MODE == 1)

// Photon mapping for caustics (-Drt.caustics): AOV_MODE 5 shoots the photons, one thread each,
// from the lights through the casters to where they land; CAUSTICS in the image mode gathers them
// at the first surface a camera path reaches that is not a caster, and the path tracer gives up
// the light it would have found there through casters, so nothing is counted twice.
#ifndef CAUSTICS
#define CAUSTICS 0
#endif
// A metal smoother than this - its fuzz, GGX's alpha for a model file's metal - bends photons;
// rougher ones are left to the path tracer, like everything that is not glass or such a metal.
#define CAUSTIC_MAX_FUZZ 0.04f
#define CAUSTIC_AIMS 8
#define PHOTON_FLOATS 12

// What a launch with caustics reads, written by gpu/CausticMap before each launch. Pointers first.
typedef struct {
    float* photons;              // a slot a photon: position, landed (1) or not (0); direction it came in, 0; flux, 0
    const float* sorted;         // the landed photons, sorted by grid cell, in the same layout
    const int* cellStart;        // tableSize + 1 entries: where each cell's run starts in sorted
    const float* lightCdf;       // a light is shot from in proportion to the light it sends the casters' way
    int photonCount;             // photons shot this launch
    int tableSize;               // grid cells (hashed), a power of two
    float cellSize;              // twice the radius: a gather disc touches at most 2 x 2 x 2 cells
    float radius;                // of the gather disc, this launch
    int aimCount;                // spheres round the casters the photons are aimed at
    int padding;
    float aim[CAUSTIC_AIMS * 4]; // centre and radius of each
} CausticParams;

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
    // 1 - z without cancellation: 1 - cos(theta max) as x / (1 + sqrt(1 - x)). Written as the book
    // writes it, 1 + r2 (sqrt(1 - x) - 1), a cone narrower than a float's precision - a 1 mm lens
    // 4 m from the sun the photons are aimed at, x = 8e-8 - collapsed to its axis: every photon
    // went through the same point of the lens.
    float x = radius * radius / distanceSquared;
    float oneMinusZ = r2 * x / (1.0f + sqrtf(fmaxf(0.0f, 1.0f - x)));
    float z = 1.0f - oneMinusZ;

    float phi = 2.0f * M_PI_F * r1;
    float root = sqrtf(fmaxf(0.0f, oneMinusZ * (2.0f - oneMinusZ)));
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
    float3 normal;                       // the shading normal, against the ray
    float3 gnormal;                      // the surface's own normal, against the ray: what is in front and behind
    float t;
    float u;
    float v;
    int material;
    bool frontFace;
#if HAS_MEDIA
    float density;                       // in fog, the volume's density: photons there are kept divided by it
#endif
#if RT_OPTIX && GUIDES_ON
    int turning;                         // the guides only: a placed mesh a turntable turns, which the flow follows
#endif
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
    const float* __restrict__ triangleNormals; // 9 per triangle: its corners' normals, zeros when flat
    const float* __restrict__ triangleUvs;     // 6 per triangle: its corners' texture coordinates, NaN first when none; null in OptiX
    const float* __restrict__ matF;            // for a normal map, found when the hit is finished
    const int* __restrict__ texI;
    const unsigned char* __restrict__ images;
} Scene;

// The closest hit found so far. Only what the search itself needs: normals, points and texture
// coordinates are worked out once, for the winner (finish_hit).
typedef struct {
    float t;
    float u;
    float v;
    int kind;                            // 0 while nothing has been hit
    int index;
#if RT_OPTIX
    int instance;                        // for LEAF_MESH: which placed mesh
#endif
} Closest;

#if RT_OPTIX

// A placed mesh: its triangles as it was loaded - nine floats each, three vertices in its own
// space - the turn of its placement about the vertical, and what it is made of. Where it
// stands is OptiX's business; shading only needs to turn a normal.
struct Instance {
    const float* vertices;
    const int* faces;                    // each triangle's entry in the mesh's own palette, or null
    const float* normals;                // 9 per triangle, its corners' normals in the mesh's space; null when flat
    const float* uvs;                    // 6 per triangle, its corners' texture coordinates; null when the file gave none
    int material;                        // the material of the mesh, and of a face not in the palette
    int palette;                         // where this instance's entries start in Params.palette
    int paletteCount;
    float cosTurn;
    float sinTurn;
    int turning;                         // 1 while a turntable turns it: the flow guide follows it
};

// What an OptiX launch is given: the scene's buffers, which the CUDA kernel takes as arguments,
// the tree OptiX searches, and the launch's numbers. gpu/optix/OptixSession fills it; its
// pointers and 64-bit numbers come first, so nothing is padded between the fields.
struct Params {
    const float* cam;
    const float4* nodes;                 // the trees of the volumes' boundaries only
    const int2* leaves;
    const int2* refs;
    const float* spheres;
    const int* sphereMaterials;
    const float4* triangles;
    const int* triangleMaterials;
    const float* quads;
    const int* quadMaterials;
    const int* mediumInts;
    const float* mediumFloats;
    const int* lights;
    const int* matI;
    const float* matF;
    const int* texI;
    const float* texF;
    const unsigned char* images;
    const float* perlinVectors;
    const int* perlinPermutations;
    const float* triangleNormals;        // the loose triangles' corner normals, as the CUDA kernel's argument
    float* out;
    float* guideNormal;                  // with GUIDES, the normal guide the albedo guide's rays find
    float* guideFlow;                    // with GUIDES, how far each pixel's first hit moved since the frame before, in pixels
    const int* quadIds;                  // the quad each pair of triangles in the tree stands for
    const int* sphereIds;                // the sphere each primitive of the spheres' tree stands for
    const Instance* instances;
    const int* palette;
    const float* env;                    // the environment map, 3 floats a pixel, or null for the background
    const CausticParams* caustic;        // with caustics, the photon map; null otherwise
    OptixTraversableHandle handle;
    unsigned long long seed;
    int width;
    int height;
    int sampleStart;
    int sampleCount;
    int maxReflectionDepth;
    float pixelSamplesScale;
    int rootNode;
    int lightCount;
    int quadInput;                       // which input of the tree of loose surfaces holds the quads; -1 without any
    float fireflyClamp;                  // the brightest a sample may be; 0 for no limit
    int mediumCount;                     // the volumes, which world_hit tests after OptiX's search
    float turnCos;                       // the turn that takes a turning mesh back to where it stood a frame
    float turnSin;                       // before, as an Instance turns (x' = c x + s z, z' = -s x + c z),
    float turnAxisX;                     // about the vertical line through (turnAxisX, turnAxisZ)
    float turnAxisZ;
    int envWidth;
    int envHeight;
    float envScale;                      // what the map's values are multiplied by
    float envTurn;                       // radians the map is turned about the vertical
};

extern "C" {
__constant__ Params params;
}

#endif

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
 * The slab test multiplies by a precomputed 1/direction, as one fused multiply-add per plane:
 * (plane - origin) * inv = fmaf(plane, inv, -origin * inv), with origin * inv worked out once per
 * traversal. NVIDIA's OpenCL compiler made that rewrite on its own under unsafe math; NVRTC does
 * not, and the two extra operations on each of the 12 planes of every node made tracing slower.
 * A ray parallel to a slab plane (inv infinite) gives inf - inf = NaN there; fminf/fmaxf return
 * their other operand for a NaN, so that axis simply stops constraining the interval instead of
 * poisoning it - a box may then be entered needlessly, never missed.
 */
#define BVH_TRAVERSE(LEAF_FN)                                                                  \
    {                                                                                          \
        float3 inv = 1.0f / direction;                                                         \
        float3 oi = origin * inv;                                                              \
        int stack[STACK_SIZE];                                                                 \
        int sp = 0;                                                                            \
        int node = root;                                                                       \
        while (true) {                                                                         \
            const float4* n = sc->nodes + node * 4;                                   \
            float4 n0 = n[0];                                                                  \
            float4 n1 = n[1];                                                                  \
            float4 n2 = n[2];                                                                  \
            float4 n3 = n[3]; int2 codes = make_int2(__float_as_int(n3.x), __float_as_int(n3.y));                                                     \
            float aLoX = fmaf(n0.x, inv.x, -oi.x), aHiX = fmaf(n0.y, inv.x, -oi.x);            \
            float aLoY = fmaf(n0.z, inv.y, -oi.y), aHiY = fmaf(n0.w, inv.y, -oi.y);            \
            float aLoZ = fmaf(n2.x, inv.z, -oi.z), aHiZ = fmaf(n2.y, inv.z, -oi.z);            \
            float bLoX = fmaf(n1.x, inv.x, -oi.x), bHiX = fmaf(n1.y, inv.x, -oi.x);            \
            float bLoY = fmaf(n1.z, inv.y, -oi.y), bHiY = fmaf(n1.w, inv.y, -oi.y);            \
            float bLoZ = fmaf(n2.z, inv.z, -oi.z), bHiZ = fmaf(n2.w, inv.z, -oi.z);            \
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

// The shading normal across a smooth facet: its corners' normals [n] (9 floats) blended by the
// hit's barycentric coordinates. It is only taken when it leans the same way as the facet
// itself, [outward]; otherwise - and for a flat facet, whose corners are zeros - the facet's own.
__device__ inline float3 smooth_normal(const float* n, float u, float v, float3 outward) {
    float w = 1.0f - u - v;
    float3 s = make_float3(n[0], n[1], n[2]) * w + make_float3(n[3], n[4], n[5]) * u + make_float3(n[6], n[7], n[8]) * v;
    float ls = length(s);
    return (ls > 1e-6f && dot(s, outward) > 0.0f) ? s / ls : outward;
}

// A mesh's texture coordinates at the hit: its corners' blended by the barycentric [u], [v] as
// smooth_normal blends normals, wrapped into [0, 1) since files tile textures past their edges.
// A triangle without coordinates (NaN first) keeps the barycentric ones.
__device__ inline void mesh_uv(const float* t, float u, float v, HitRecord* rec) {
    if (isnan(t[0])) return;
    float w = 1.0f - u - v;
    float tu = t[0] * w + t[2] * u + t[4] * v;
    float tv = t[1] * w + t[3] * u + t[5] * v;
    rec->u = tu - floorf(tu);
    rec->v = tv - floorf(tv);
}

// What a ray that escapes sees: the environment map when there is one - latitude-longitude, its
// rows from straight up to straight down, turned [turn] radians about the vertical - else the
// background colour, else the first book's sky gradient.
__device__ inline float3 sky_colour(const float* env, int w, int h, float scale, float turn,
                                    float3 direction, float3 background, bool flatBackground) {
    float3 d = direction / length(direction);
    if (env) {
        float u = (atan2f(d.x, -d.z) + turn) * (0.5f / M_PI_F) + 0.5f;
        u -= floorf(u);
        float v = acosf(clamp(d.y, -1.0f, 1.0f)) * (1.0f / M_PI_F);
        int i = min((int)(u * (float)w), w - 1);
        int j = min((int)(v * (float)h), h - 1);
        const float* p = env + (j * w + i) * 3;
        return make_float3(p[0], p[1], p[2]) * scale;
    }
    if (flatBackground) return background;
    float alpha = 0.5f * (d.y + 1.0f);
    return make_float3(1.0f, 1.0f, 1.0f) * (1.0f - alpha) + make_float3(0.5f, 0.7f, 1.0f) * alpha;
}

// The directions a triangle's texture u and v grow in, from its edges [e1], [e2] and its corners'
// coordinates [t]: what a normal map's red and green lean the normal towards. False without
// coordinates, or when they do not span the triangle.
__device__ inline bool uv_tangents(float3 e1, float3 e2, const float* t, float3* tu, float3* tv) {
    if (isnan(t[0])) return false;
    float du1 = t[2] - t[0], dv1 = t[3] - t[1], du2 = t[4] - t[0], dv2 = t[5] - t[1];
    float det = du1 * dv2 - du2 * dv1;
    if (fabsf(det) < 1e-12f) return false;
    float r = 1.0f / det;
    *tu = (e1 * dv2 - e2 * dv1) * r;
    *tv = (e2 * du1 - e1 * du2) * r;
    return true;
}

// One texel of an image texture as stored, 0..1 - for a normal map, whose bytes are data.
__device__ inline float3 image_rgb(const int* texI, const unsigned char* images, int index, float u, float v) {
    if (texI[index * 4] != 2) return make_float3(0.5f, 0.5f, 1.0f);
    int offset = texI[index * 4 + 1];
    int width = texI[index * 4 + 2];
    int height = texI[index * 4 + 3];
    int i = min(max((int)(clamp(u, 0.0f, 1.0f) * (float)width), 0), width - 1);
    int j = min(max((int)((1.0f - clamp(v, 0.0f, 1.0f)) * (float)height), 0), height - 1);
    int at = offset + (j * width + i) * 3;
    return make_float3((float)images[at], (float)images[at + 1], (float)images[at + 2]) * (1.0f / 255.0f);
}

// Where [p] lands on the screen of a pinhole camera at [center] whose pixel (0, 0) is centred at
// [pixel00] and whose pixels step by [du] and [dv]: (column, row), in pixels, fractions kept.
__device__ inline float2 screen_position(float3 p, float3 center, float3 pixel00, float3 du, float3 dv) {
    float3 n = cross(du, dv);
    float3 d = p - center;
    float3 q = center + d * (dot(pixel00 - center, n) / dot(d, n)) - pixel00;
    return make_float2(dot(q, du) / dot(du, du), dot(q, dv) / dot(dv, dv));
}

// Everything shading needs about the winning hit, computed once.
__device__ inline void finish_hit(const Scene* sc, const Closest* best, float3 origin, float3 direction,
                       float time, HitRecord* rec) {
#if RT_OPTIX && GUIDES_ON
    rec->turning = 0;
#endif
    rec->t = best->t;
    rec->point = origin + best->t * direction;
    rec->u = best->u;
    rec->v = best->v;
    float3 outwardNormal;
    float3 shading;
    // The texture's u and v directions, for a normal map; only a textured triangle has them.
    float3 tanU = make_float3(0.0f, 0.0f, 0.0f);
    float3 tanV = make_float3(0.0f, 0.0f, 0.0f);
    bool hasTangents = false;

    if (HAS_SPHERES && best->kind == LEAF_SPHERE) {
        const float* s = sc->spheres + best->index * 8;
        float3 center = make_float3(s[0], s[1], s[2]);
    if (HAS_MOTION) center += time * make_float3(s[3], s[4], s[5]);
        outwardNormal = (rec->point - center) / s[6];
        sphere_uv(outwardNormal, rec);
        rec->material = sc->sphereMaterials[best->index];
        shading = outwardNormal;
    } else if (HAS_TRIANGLES && best->kind == LEAF_TRIANGLE) {
        const float4* t = sc->triangles + best->index * 3;
        outwardNormal = cross(xyz(t[1]), xyz(t[2]));
        outwardNormal = outwardNormal / length(outwardNormal);
        rec->material = sc->triangleMaterials[best->index];
        shading = HAS_SMOOTH ? smooth_normal(sc->triangleNormals + best->index * 9, best->u, best->v, outwardNormal) : outwardNormal;
        if (HAS_UV && sc->triangleUvs) {
            if (HAS_NORMALMAP) hasTangents = uv_tangents(xyz(t[1]), xyz(t[2]), sc->triangleUvs + best->index * 6, &tanU, &tanV);
            mesh_uv(sc->triangleUvs + best->index * 6, best->u, best->v, rec);
        }
    } else if (HAS_QUADS && best->kind == LEAF_QUAD) {
        const float* data = sc->quads + best->index * 16;
        outwardNormal = make_float3(data[12], data[13], data[14]);
        rec->material = sc->quadMaterials[best->index];
        shading = outwardNormal;
#if RT_OPTIX
    } else if (best->kind == LEAF_MESH) {
        // The facet's normal from its winding, in the mesh's own space, then turned as the
        // placement turns the mesh; its scale and offset leave a direction alone. A smooth
        // mesh's normal is blended in its own space and turned the same way.
        const Instance* placed = params.instances + best->instance;
        const float* v = placed->vertices + best->index * 9;
        float3 v0 = make_float3(v[0], v[1], v[2]);
        float3 facet = cross(make_float3(v[3], v[4], v[5]) - v0, make_float3(v[6], v[7], v[8]) - v0);
        facet = facet / length(facet);
        float3 n = (HAS_SMOOTH && placed->normals) ? smooth_normal(placed->normals + best->index * 9, best->u, best->v, facet) : facet;
        if (HAS_UV && placed->uvs) {
            if (HAS_NORMALMAP && uv_tangents(make_float3(v[3], v[4], v[5]) - v0, make_float3(v[6], v[7], v[8]) - v0,
                                             placed->uvs + best->index * 6, &tanU, &tanV)) {
                // Turned as the placement turns the mesh, like its normals.
                tanU = make_float3(placed->cosTurn * tanU.x + placed->sinTurn * tanU.z, tanU.y, -placed->sinTurn * tanU.x + placed->cosTurn * tanU.z);
                tanV = make_float3(placed->cosTurn * tanV.x + placed->sinTurn * tanV.z, tanV.y, -placed->sinTurn * tanV.x + placed->cosTurn * tanV.z);
                hasTangents = true;
            }
            mesh_uv(placed->uvs + best->index * 6, best->u, best->v, rec);
        }
        outwardNormal = make_float3(placed->cosTurn * facet.x + placed->sinTurn * facet.z, facet.y,
                                    -placed->sinTurn * facet.x + placed->cosTurn * facet.z);
        shading = make_float3(placed->cosTurn * n.x + placed->sinTurn * n.z, n.y, -placed->sinTurn * n.x + placed->cosTurn * n.z);
        rec->material = placed->material;
#if GUIDES_ON
        rec->turning = placed->turning;
#endif
        if (placed->faces) {
            int entry = placed->faces[best->index];
            if (entry >= 0 && entry < placed->paletteCount) rec->material = params.palette[placed->palette + entry];
        }
#endif
    } else {
        // Fog has no surface: the normal is arbitrary and the ray always counts as outside.
        rec->normal = make_float3(1.0f, 0.0f, 0.0f);
        rec->gnormal = rec->normal;
        rec->frontFace = true;
        rec->material = sc->mediumInts[best->index * 2 + 1];
        rec->u = 0.0f;
        rec->v = 0.0f;
#if HAS_MEDIA
        rec->density = -1.0f / sc->mediumFloats[best->index];
#endif
        return;
    }
    // In front or behind is the surface's own business; the shading normal only shades.
    rec->frontFace = dot(direction, outwardNormal) < 0.0f;
    rec->gnormal = rec->frontFace ? outwardNormal : -outwardNormal;
    rec->normal = rec->frontFace ? shading : -shading;
    // A shading normal turned away from the viewer would shade the back of the facet.
    if (dot(rec->normal, direction) > 0.0f) rec->normal = rec->gnormal;
#if HAS_NORMALMAP
    // A normal map (the material's fourth float, its texture plus one): red and green lean the
    // normal along the texture's u and v, DirectX-style with green pointing down v; the blue is
    // worked out from them, as two-channel maps leave it out.
    int normalMap = hasTangents ? (int)sc->matF[rec->material * 4 + 3] - 1 : -1;
    if (normalMap >= 0) {
        float3 c = image_rgb(sc->texI, sc->images, normalMap, rec->u, rec->v);
        float nx = c.x * 2.0f - 1.0f;
        float ny = -(c.y * 2.0f - 1.0f);
        float nz = sqrtf(fmaxf(0.0f, 1.0f - nx * nx - ny * ny));
        float3 n = rec->frontFace ? rec->normal : -rec->normal;
        float3 t = tanU - n * dot(n, tanU);
        float lt = length(t);
        if (lt > 1e-8f) {
            t = t / lt;
            float3 b = cross(n, t);
            if (dot(b, tanV) < 0.0f) b = -b;
            float3 p = t * nx + b * ny + n * nz;
            p = p / length(p);
            float3 shaded = rec->frontFace ? p : -p;
            if (dot(shaded, direction) < 0.0f) rec->normal = shaded;
        }
    }
#endif
}

#if RT_OPTIX

#define OPTIX_RAY_END 1e16f
// Whether the ray's time rides along in a seventh payload value, for the spheres' intersection
// program to move moving spheres by. Only a scene with moving spheres pays for the extra value:
// declared in every scene, it slowed Yennefer by 4%.
#ifndef OPTIX_RAY_TIME
#define OPTIX_RAY_TIME 0
#endif

// The closest hit, asked of OptiX. It comes back in the ray's payload as the closest-hit
// programs at the end of this file fill it: what was hit, where along the ray, a triangle's
// barycentric coordinates - which are the u/v triangle_t reports - and, for a placed mesh,
// which one.
__device__ inline bool world_hit(const Scene* sc, int root, float3 origin, float3 direction, float time,
                      float tMin, float tMax, Rng* rng, HitRecord* rec) {
    unsigned int kind = 0u, index = 0u, t = 0u, u = 0u, v = 0u, instance = 0u;
    // OptiX takes a ray reaching to infinity as invalid (its validation mode stops on it), so the
    // search ends at a distance no scene comes near.
#if OPTIX_RAY_TIME
    unsigned int rayTime = __float_as_uint(time);
    optixTrace(params.handle, origin, direction, tMin, fminf(tMax, OPTIX_RAY_END), 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0u, 1u, 0u, kind, index, t, u, v, instance, rayTime);
#else
    optixTrace(params.handle, origin, direction, tMin, fminf(tMax, OPTIX_RAY_END), 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0u, 1u, 0u, kind, index, t, u, v, instance);
#endif
    // Fog: OptiX found the nearest surface, and a volume may stop the ray before it. Its
    // boundary is a small tree medium_t searches itself, and a scene has a volume or two.
    if (HAS_MEDIA) {
        float nearest = kind != 0u ? __uint_as_float(t) : tMax;
        for (int m = 0; m < params.mediumCount; m++) {
            float inside;
            if (medium_t(sc, m, origin, direction, time, tMin, nearest, rng, &inside)) {
                nearest = inside;
                kind = (unsigned int)LEAF_MEDIUM;
                index = (unsigned int)m;
                t = __float_as_uint(inside);
            }
        }
    }
    if (kind == 0u) return false;

    Closest found;
    found.kind = (int)kind;
    found.index = (int)index;
    found.instance = (int)instance;
    found.t = __uint_as_float(t);
    found.u = __uint_as_float(u);
    found.v = __uint_as_float(v);
    if (HAS_QUADS && found.kind == LEAF_QUAD) {
        // OptiX sees a quad as two triangles; its own u/v come from the hit point, as in quad_t.
        const float* data = sc->quads + found.index * 16;
        float3 Q = make_float3(data[0], data[1], data[2]);
        float3 edgeU = make_float3(data[3], data[4], data[5]);
        float3 edgeV = make_float3(data[6], data[7], data[8]);
        float3 w = make_float3(data[9], data[10], data[11]);
        float3 planarHit = origin + found.t * direction - Q;
        found.u = dot(w, cross(planarHit, edgeV));
        found.v = dot(w, cross(edgeU, planarHit));
    }
    finish_hit(sc, &found, origin, direction, time, rec);
    return true;
}

#else

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

#endif

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
__device__ inline float3 material_emitted(const int* matI, const float* matF,
                               const int* texI, const float* texF,
                               const unsigned char* images,
                               const float* perlinVectors, const int* perlinPermutations,
                               const HitRecord* rec) {
    // Only emissive materials, and only through their front: a lamp does not light the room
    // through the back of the ceiling.
    if (!MAT_EMISSIVE || !rec->frontFace) return make_float3(0.0f, 0.0f, 0.0f);
    int m = rec->material * 4;
    if (matI[m] == 3)
        return texture_value(texI, texF, images, perlinVectors, perlinPermutations, matI[m + 1], rec->u, rec->v, rec->point);
    // A surface that also glows where its emission texture says so - embers in a bowl.
    if (matI[m + 2] >= 0)
        return texture_value(texI, texF, images, perlinVectors, perlinPermutations, matI[m + 2], rec->u, rec->v, rec->point) * matF[m + 2];
    return make_float3(0.0f, 0.0f, 0.0f);
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
    int type = matI[rec->material * 4];
    int texture = matI[rec->material * 4 + 1];

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
        float fuzz = matF[rec->material * 4];
        // A roughness texture, read as data: GGX's alpha, its square, blurs the reflection.
        int roughTexture = matI[rec->material * 4 + 3];
        if (roughTexture >= 0) {
            float r = texture_value(texI, texF, images, perlinVectors, perlinPermutations, roughTexture, rec->u, rec->v, rec->point).x;
            fuzz = r * r;
        }
        float3 tint = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                    texture, rec->u, rec->v, rec->point);
        if (matF[rec->material * 4 + 1] > 0.5f) {
            // A model file's metal, as a game engine draws it: GGX microfacets of alpha = roughness
            // squared (the fuzz above), a facet normal drawn in proportion to D(h) cos(h), the
            // reflection off it weighed by Schlick's Fresnel and Smith's shadowing,
            // F G (v.h) / ((n.v)(n.h)) - the rest of the microfacet density cancels in that draw.
            float alpha = fmaxf(fuzz, 0.002f);
            float a2 = alpha * alpha;
            float3 n = rec->normal;
            float3 v = -unitDirection;
            float nv = dot(n, v);
            if (nv <= 0.0f) return false;
            float u1 = rng_float(rng);
            float u2 = rng_float(rng);
            float cos2 = (1.0f - u2) / (1.0f + (a2 - 1.0f) * u2);
            float cosH = sqrtf(cos2);
            float sinH = sqrtf(fmaxf(0.0f, 1.0f - cos2));
            float phi = 2.0f * M_PI_F * u1;
            float3 bu, bv, bw;
            onb_build(n, &bu, &bv, &bw);
            float3 h = bu * (sinH * cosf(phi)) + bv * (sinH * sinf(phi)) + bw * cosH;
            float vh = dot(v, h);
            if (vh <= 0.0f) return false;
            float3 l = h * (2.0f * vh) - v;
            float nl = dot(n, l);
            if (nl <= 0.0f || dot(l, rec->gnormal) <= 0.0f) return false;
            float m = 1.0f - vh;
            float m5 = m * m * m * m * m;
            float3 fresnel = tint + (make_float3(1.0f, 1.0f, 1.0f) - tint) * m5;
            float gv = 2.0f * nv / (nv + sqrtf(a2 + (1.0f - a2) * nv * nv));
            float gl = 2.0f * nl / (nl + sqrtf(a2 + (1.0f - a2) * nl * nl));
            srec->attenuation = fresnel * (gv * gl * vh / (nv * cosH));
            srec->direction = l;
            return true;
        }
        float3 reflected = unitDirection - rec->normal * (2.0f * dot(unitDirection, rec->normal));
        reflected = reflected / length(reflected);
        srec->attenuation = tint;
        srec->direction = reflected + random_unit_vector(rng) * fuzz;
        // A reflection under the surface itself - off a smooth normal, or fuzzed past the
        // horizon - is absorbed, as in the book's Metal.
        return dot(srec->direction, rec->gnormal) > 0.0f;
    }

    // Dielectric - the only material left, unless the scene has none.
    if (!MAT_DIELECTRIC) return false;
    float ri = matF[rec->material * 4 + 1];
    float refractionRatio = rec->frontFace ? (1.0f / ri) : ri;
    float cosTheta = fminf(dot(-unitDirection, rec->normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    bool cannotRefract = refractionRatio * sinTheta > 1.0f;
    // Short-circuit: the random draw only happens when refraction is possible, as on the CPU.
    bool useReflect = cannotRefract || (reflectance(cosTheta, refractionRatio) > rng_float(rng));

    srec->attenuation = make_float3(1.0f, 1.0f, 1.0f);
    // Tinted glass (Beer-Lambert): a ray leaving it - a hit on the inside - has gone rec->t times
    // its direction's length through it, and [tint distance] of that leaves the tint's colour.
    float tintDistance = matF[rec->material * 4];
    if (tintDistance > 0.0f && !rec->frontFace) {
        float3 tint = texture_value(texI, texF, images, perlinVectors, perlinPermutations, texture, rec->u, rec->v, rec->point);
        float k = rec->t * length(inDirection) / tintDistance;
        srec->attenuation = make_float3(powf(tint.x, k), powf(tint.y, k), powf(tint.z, k));
    }
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
    int type = matI[rec->material * 4];
    if (type == 2) return make_float3(1.0f, 1.0f, 1.0f);
    float3 colour = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                  matI[rec->material * 4 + 1], rec->u, rec->v, rec->point);
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
    int type = matI[rec->material * 4];
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

// ---------------------------------------------------------------- caustics

/**
 * Whether light passing through this surface makes caustics the photons carry: glass, and metal
 * smoother than CAUSTIC_MAX_FUZZ where it was hit (its roughness texture read there). The one
 * test both the photon pass and the camera paths make, so they cannot disagree on who counts what.
 */
__device__ inline bool is_caster(const int* matI, const float* matF, const int* texI, const float* texF,
                                 const unsigned char* images, const float* perlinVectors, const int* perlinPermutations,
                                 const HitRecord* rec) {
    int m = rec->material * 4;
    int kind = matI[m];
    if (kind == 2) return true;
    if (kind != 1) return false;
    float fuzz = matF[m];
    int rough = matI[m + 3];
    if (rough >= 0) {
        float r = texture_value(texI, texF, images, perlinVectors, perlinPermutations, rough, rec->u, rec->v, rec->point).x;
        fuzz = r * r;
    }
    return fuzz <= CAUSTIC_MAX_FUZZ;
}

/** The grid cell a point falls in, hashed into the table; kernels/caustics_grid.cu hashes the same way. */
__device__ inline int caustic_cell(int x, int y, int z, int mask) {
    unsigned int h = ((unsigned int)x * 73856093u) ^ ((unsigned int)y * 19349663u) ^ ((unsigned int)z * 83492791u);
    return (int)(h & (unsigned int)mask);
}

#if CAUSTICS
/**
 * The irradiance the photons bring to point x, whose normal n faces the camera's path: the flux
 * of those in a disc of the launch's radius, on the surface (a thin slab round its plane) and
 * arriving on the path's side, weighed by 1 - d^2/r^2 (Epanechnikov) over that kernel's integral,
 * pi r^2 / 2. A Lambertian surface turns it into radiance with albedo / pi.
 */
__device__ inline float3 caustic_gather(const CausticParams* cp, float3 x, float3 n) {
    float r = cp->radius;
    float r2 = r * r;
    float cs = cp->cellSize;
    float slab = 0.25f * r;
    int mask = cp->tableSize - 1;
    int x0 = (int)floorf((x.x - r) / cs), x1 = (int)floorf((x.x + r) / cs);
    int y0 = (int)floorf((x.y - r) / cs), y1 = (int)floorf((x.y + r) / cs);
    int z0 = (int)floorf((x.z - r) / cs), z1 = (int)floorf((x.z + r) / cs);
    int seen[8];
    int seenCount = 0;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int cz = z0; cz <= z1; cz++)
        for (int cy = y0; cy <= y1; cy++)
            for (int cx = x0; cx <= x1; cx++) {
                int h = caustic_cell(cx, cy, cz, mask);
                // Two cells that hash alike share a run: read it once.
                bool again = false;
                for (int k = 0; k < seenCount; k++) again = again || seen[k] == h;
                if (again) continue;
                if (seenCount < 8) seen[seenCount++] = h;
                int end = cp->cellStart[h + 1];
                for (int i = cp->cellStart[h]; i < end; i++) {
                    const float* P = cp->sorted + i * PHOTON_FLOATS;
                    float3 d = make_float3(P[0] - x.x, P[1] - x.y, P[2] - x.z);
                    float d2 = dot(d, d);
                    if (d2 >= r2) continue;
                    if (P[7] != 0.0f) continue;          // one kept in fog
                    if (fabsf(dot(d, n)) > slab) continue;
                    if (P[4] * n.x + P[5] * n.y + P[6] * n.z >= 0.0f) continue;
                    sum += make_float3(P[8], P[9], P[10]) * (1.0f - d2 / r2);
                }
            }
    return sum * (2.0f / (M_PI_F * r2));
}

#if HAS_MEDIA
/**
 * The fluence (the radiance summed over all directions) the photons bring to point x in fog: the
 * photons kept in fog in a ball of the launch's radius, each already divided by the density of
 * the fog it stopped in - a photon stops in a stretch ds with chance density * ds, so the flux
 * kept in a volume V is density * fluence * V - weighed by 1 - d^2/r^2 over that kernel's
 * integral in three dimensions, 8 pi r^3 / 15. Isotropic fog turns it into radiance with
 * albedo / (4 pi). The beam of light a lens sends through smoke, which the camera's paths alone
 * would almost never find: from a point in the beam, only the directions back along it through
 * the lens lead to the lamp.
 */
__device__ inline float3 caustic_gather_volume(const CausticParams* cp, float3 x) {
    float r = cp->radius;
    float r2 = r * r;
    float cs = cp->cellSize;
    int mask = cp->tableSize - 1;
    int x0 = (int)floorf((x.x - r) / cs), x1 = (int)floorf((x.x + r) / cs);
    int y0 = (int)floorf((x.y - r) / cs), y1 = (int)floorf((x.y + r) / cs);
    int z0 = (int)floorf((x.z - r) / cs), z1 = (int)floorf((x.z + r) / cs);
    int seen[8];
    int seenCount = 0;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int cz = z0; cz <= z1; cz++)
        for (int cy = y0; cy <= y1; cy++)
            for (int cx = x0; cx <= x1; cx++) {
                int h = caustic_cell(cx, cy, cz, mask);
                bool again = false;
                for (int k = 0; k < seenCount; k++) again = again || seen[k] == h;
                if (again) continue;
                if (seenCount < 8) seen[seenCount++] = h;
                int end = cp->cellStart[h + 1];
                for (int i = cp->cellStart[h]; i < end; i++) {
                    const float* P = cp->sorted + i * PHOTON_FLOATS;
                    if (P[7] == 0.0f) continue;          // one on a surface
                    float3 d = make_float3(P[0] - x.x, P[1] - x.y, P[2] - x.z);
                    float d2 = dot(d, d);
                    if (d2 >= r2) continue;
                    sum += make_float3(P[8], P[9], P[10]) * (1.0f - d2 / r2);
                }
            }
    return sum * (15.0f / (8.0f * M_PI_F * r2 * r));
}
#endif
#endif

#if AOV_MODE == 5
/**
 * Photon [index] of the launch: from a light picked in proportion to the light it sends the
 * casters, from a point on it, in a direction aimed at one of the spheres round the casters (the
 * density being the mixture over all of them, right where they overlap); refracted and reflected
 * by each caster as a camera path would be, until it lands on a Lambertian surface - stored, with
 * its flux - or is lost: missed everything, landed before any caster, on a lamp or on anything
 * else. Its slot says which.
 */
__device__ void photon_shoot(const Scene* sc, int rootNode,
                             const int* matI, const float* matF, const int* texI, const float* texF,
                             const unsigned char* images, const float* perlinVectors, const int* perlinPermutations,
                             const int* lights, int lightCount, const CausticParams* cp, int index,
                             unsigned long long seed, int sampleStart, int maxDepth) {
    float* slot = cp->photons + index * PHOTON_FLOATS;
    slot[3] = 0.0f;
    if (lightCount <= 0 || cp->aimCount <= 0) return;
    Rng rng;
    // Streams of their own, apart from every pixel's.
    rng_init(&rng, seed ^ 0x5DEECE66DULL, 0x100000000ULL + (unsigned long long)index, (unsigned long long)sampleStart);

    float pickAt = rng_float(&rng);
    int pick = 0;
    while (pick < lightCount - 1 && cp->lightCdf[pick] < pickAt) pick++;
    float prob = cp->lightCdf[pick] - (pick > 0 ? cp->lightCdf[pick - 1] : 0.0f);
    if (prob <= 0.0f) return;

    int kind = lights[pick * 2];
    int li = lights[pick * 2 + 1];
    float3 p, n;
    float pdfArea, extent;
    int material;
    if (kind == LEAF_QUAD) {
        const float* q = sc->quads + li * 16;
        float3 Q = make_float3(q[0], q[1], q[2]);
        float3 U = make_float3(q[3], q[4], q[5]);
        float3 V = make_float3(q[6], q[7], q[8]);
        p = Q + U * rng_float(&rng) + V * rng_float(&rng);
        n = make_float3(q[12], q[13], q[14]);          // a quad lights only its front
        float area = length(cross(U, V));
        pdfArea = 1.0f / area;
        extent = sqrtf(area);
        material = sc->quadMaterials[li];
    } else {
        const float* sp = sc->spheres + li * 8;
        float radius = sp[6];
        n = random_unit_vector(&rng);
        p = make_float3(sp[0], sp[1], sp[2]) + n * radius;
        pdfArea = 1.0f / (4.0f * M_PI_F * radius * radius);
        extent = radius;
        material = sc->sphereMaterials[li];
    }
    float3 emitted = texture_value(texI, texF, images, perlinVectors, perlinPermutations, matI[material * 4 + 1], 0.5f, 0.5f, p);

    int a = min((int)(rng_float(&rng) * (float)cp->aimCount), cp->aimCount - 1);
    float3 toAim = make_float3(cp->aim[a * 4], cp->aim[a * 4 + 1], cp->aim[a * 4 + 2]) - p;
    float aimRadius = cp->aim[a * 4 + 3];
    float aimDistance2 = dot(toAim, toAim);
    float3 d;
    if (aimDistance2 <= aimRadius * aimRadius) {
        d = random_unit_vector(&rng);
    } else {
        float3 bu, bv, bw;
        onb_build(toAim, &bu, &bv, &bw);
        float3 s = random_to_sphere(&rng, aimRadius, aimDistance2);
        d = bu * s.x + bv * s.y + bw * s.z;
    }
    d = d / length(d);
    float cosLight = dot(n, d);
    if (cosLight <= 0.0f) return;

    // The density of that direction: the mixture over every aim's cone that holds it.
    float pdfDirection = 0.0f;
    for (int k = 0; k < cp->aimCount; k++) {
        float3 t = make_float3(cp->aim[k * 4], cp->aim[k * 4 + 1], cp->aim[k * 4 + 2]) - p;
        float tt = dot(t, t);
        float rk2 = cp->aim[k * 4 + 3] * cp->aim[k * 4 + 3];
        if (tt <= rk2) { pdfDirection += 1.0f / (4.0f * M_PI_F); continue; }
        float x = rk2 / tt;
        float cosMax = sqrtf(1.0f - x);
        // 1 - cosMax as x / (1 + cosMax): no cancellation for a far, small aim (the sun 3400 units off).
        if (dot(d, t) * rsqrtf(tt) >= cosMax) pdfDirection += 1.0f / (2.0f * M_PI_F * (x / (1.0f + cosMax)));
    }
    pdfDirection /= (float)cp->aimCount;
    if (pdfDirection <= 0.0f) return;

    float3 flux = emitted * (cosLight / (pdfArea * pdfDirection * prob * (float)cp->photonCount));
    float3 origin = p + n * (extent * 1e-5f + 1e-4f);
    float time = rng_float(&rng);
    int casterHits = 0;
    for (int bounce = 0; bounce < maxDepth; bounce++) {
        HitRecord rec;
        if (!world_hit(sc, rootNode, origin, d, time, 0.001f, INFINITY, &rng, &rec)) return;
        int hitKind = matI[rec.material * 4];
        if (hitKind == 3) return;
        if (is_caster(matI, matF, texI, texF, images, perlinVectors, perlinPermutations, &rec)) {
            ScatterRec srec;
            if (!scatter(matI, matF, texI, texF, images, perlinVectors, perlinPermutations, &rec, d, &rng, &srec)) return;
            flux *= srec.attenuation;
            origin = rec.point;
            d = srec.direction / length(srec.direction);
            casterHits++;
            continue;
        }
        if (casterHits == 0) return;
#if HAS_MEDIA
        if (hitKind == 4) {
            // Stopped in fog past a caster: kept there, divided by the fog's density (caustic_gather_volume).
            flux = flux * (1.0f / rec.density);
            slot[0] = rec.point.x; slot[1] = rec.point.y; slot[2] = rec.point.z; slot[3] = 1.0f;
            slot[4] = d.x; slot[5] = d.y; slot[6] = d.z; slot[7] = 1.0f;
            slot[8] = flux.x; slot[9] = flux.y; slot[10] = flux.z; slot[11] = 0.0f;
            return;
        }
#endif
        if (hitKind != 0) return;
        slot[0] = rec.point.x; slot[1] = rec.point.y; slot[2] = rec.point.z; slot[3] = 1.0f;
        slot[4] = d.x; slot[5] = d.y; slot[6] = d.z; slot[7] = 0.0f;
        slot[8] = flux.x; slot[9] = flux.y; slot[10] = flux.z; slot[11] = 0.0f;
        return;
    }
}
#endif

// ---------------------------------------------------------------- entry point

#if RT_OPTIX

// The hit OptiX settled on, for world_hit to read back: a triangle of a placed mesh...
extern "C" __global__ void __closesthit__mesh() {
    float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0((unsigned int)LEAF_MESH);
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));
    optixSetPayload_5(optixGetInstanceId());
}

// ...or one of the scene's loose triangles or quads.
extern "C" __global__ void __closesthit__surface() {
    unsigned int primitive = optixGetPrimitiveIndex();
    float2 barycentrics = optixGetTriangleBarycentrics();
    if ((int)optixGetSbtGASIndex() == params.quadInput) {
        optixSetPayload_0((unsigned int)LEAF_QUAD);
        optixSetPayload_1((unsigned int)params.quadIds[primitive >> 1]);
    } else {
        optixSetPayload_0((unsigned int)LEAF_TRIANGLE);
        optixSetPayload_1(primitive);
    }
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));
}

// ...or a sphere, which OptiX knows only by its box: whether and where the ray meets it is
// sphere_t's to say, the same test the CUDA kernel makes. Its tree is not an instance's, so the
// ray is already in the scene's space; the ray's time moves a moving sphere.
extern "C" __global__ void __intersection__sphere() {
    Scene sc;
    sc.spheres = params.spheres;
#if OPTIX_RAY_TIME
    float time = __uint_as_float(optixGetPayload_6());
#else
    float time = 0.0f;
#endif
    float t;
    if (sphere_t(&sc, params.sphereIds[optixGetPrimitiveIndex()], optixGetWorldRayOrigin(), optixGetWorldRayDirection(),
                 time, optixGetRayTmin(), optixGetRayTmax(), &t)) {
        optixReportIntersection(t, 0u);
    }
}

extern "C" __global__ void __closesthit__sphere() {
    optixSetPayload_0((unsigned int)LEAF_SPHERE);
    optixSetPayload_1((unsigned int)params.sphereIds[optixGetPrimitiveIndex()]);
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_3(0u);
    optixSetPayload_4(0u);
}

extern "C" __global__ void __miss__nothing() {
    optixSetPayload_0(0u);
}

// One launch index is one pixel. The names below are the CUDA kernel's arguments, so the body
// that follows is the same text for both.
extern "C" __global__ void __raygen__render() {
    const float* cam = params.cam;
    const float4* nodes = params.nodes;
    const int2* leaves = params.leaves;
    const int2* refs = params.refs;
    const float* spheres = params.spheres;
    const int* sphereMaterials = params.sphereMaterials;
    const float4* triangles = params.triangles;
    const int* triangleMaterials = params.triangleMaterials;
    const float* quads = params.quads;
    const int* quadMaterials = params.quadMaterials;
    const int* mediumInts = params.mediumInts;
    const float* mediumFloats = params.mediumFloats;
    const int* lights = params.lights;
    const int* matI = params.matI;
    const float* matF = params.matF;
    const int* texI = params.texI;
    const float* texF = params.texF;
    const unsigned char* images = params.images;
    const float* perlinVectors = params.perlinVectors;
    const int* perlinPermutations = params.perlinPermutations;
    const float* triangleNormals = params.triangleNormals;
    const float* triangleUvs = nullptr;  // the loose triangles never carry texture coordinates under OptiX
    const float fireflyClamp = params.fireflyClamp;
    const float* env = params.env;
    const int envWidth = params.envWidth;
    const int envHeight = params.envHeight;
    const float envScale = params.envScale;
    const float envTurn = params.envTurn;
    const CausticParams* caustic = params.caustic;
    float* out = params.out;
    float* guideNormal = params.guideNormal;
    float* guideFlow = params.guideFlow;
    const int width = params.width;
    const int height = params.height;
    const int sampleStart = params.sampleStart;
    const int sampleCount = params.sampleCount;
    const int maxReflectionDepth = params.maxReflectionDepth;
    const float pixelSamplesScale = params.pixelSamplesScale;
    const int rootNode = params.rootNode;
    const int lightCount = params.lightCount;
    const unsigned long long seed = params.seed;

    const uint3 launchIndex = optixGetLaunchIndex();
    int x = (int)launchIndex.x;
    int y = (int)launchIndex.y;

#else

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
                     const float* __restrict__ triangleNormals,
                     const float* __restrict__ triangleUvs,
                     float* out,
                     float* guideNormal,
                     float* guideFlow,
                     const int width,
                     const int height,
                     const int sampleStart,
                     const int sampleCount,
                     const int maxReflectionDepth,
                     const float pixelSamplesScale,
                     const int rootNode,
                     const int lightCount,
                     const unsigned long long seed,
                     const float fireflyClamp,
                     const float* __restrict__ env,
                     const int envWidth,
                     const int envHeight,
                     const float envScale,
                     const float envTurn,
                     const CausticParams* __restrict__ caustic) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
#if AOV_MODE == 5
    // The photon pass: a thread a photon, the "image" photonCount wide and one high.
    int x = gid;
    int y = 0;
#else
    // Work items come in warps of 32, each covering an 8x4 tile of pixels.
    int tile = gid >> 5;
    int lane = gid & 31;
    int tilesX = (width + 7) >> 3;
    int x = (tile % tilesX) * 8 + (lane & 7);
    int y = (tile / tilesX) * 4 + (lane >> 3);
#endif
    if (x >= width || y >= height) return;

#endif

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
    scene.triangleNormals = triangleNormals;
    scene.triangleUvs = triangleUvs;
    scene.matF = matF;
    scene.texI = texI;
    scene.images = images;

#if AOV_MODE == 5
    photon_shoot(&scene, rootNode, matI, matF, texI, texF, images, perlinVectors, perlinPermutations,
                 lights, lightCount, caustic, x, seed, sampleStart, maxReflectionDepth);
    return;
#endif

    float3 cameraCenter = make_float3(cam[0], cam[1], cam[2]);
    float3 pixel00      = make_float3(cam[3], cam[4], cam[5]);
    float3 pixelDeltaU  = make_float3(cam[6], cam[7], cam[8]);
    float3 pixelDeltaV  = make_float3(cam[9], cam[10], cam[11]);
    float3 defocusDiscU = make_float3(cam[12], cam[13], cam[14]);
    float3 defocusDiscV = make_float3(cam[15], cam[16], cam[17]);
    float defocusAngle  = cam[18];
    bool flatBackground = cam[19] > 0.5f;
    float3 background   = make_float3(cam[20], cam[21], cam[22]);
#if GUIDES_ON
    // Where the camera stood for the frame before, for the flow guide: the same as now in a film.
    float3 previousCenter  = make_float3(cam[23], cam[24], cam[25]);
    float3 previousPixel00 = make_float3(cam[26], cam[27], cam[28]);
    float3 previousDeltaU  = make_float3(cam[29], cam[30], cam[31]);
    float3 previousDeltaV  = make_float3(cam[32], cam[33], cam[34]);
#endif

    float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
    float3 normalSum = make_float3(0.0f, 0.0f, 0.0f);
    float flowX = 0.0f, flowY = 0.0f;

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
#if CAUSTICS
    // 0 until the path reaches a surface that is not a caster; 1 when that was a gathered
    // Lambertian, 2 once a caster follows it - a lamp found then is the photons' to count -
    // and 3 once the path is past all that.
    int causticState = 0;
    float3 causticColor = make_float3(0.0f, 0.0f, 0.0f);
#endif

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
#if CAUSTICS
            causticState = 0;
            causticColor = make_float3(0.0f, 0.0f, 0.0f);
#endif
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
                    float3 sky = sky_colour(env, envWidth, envHeight, envScale, envTurn, direction, background, flatBackground);
                    shaded += carried * sky;
                    break;
                }
                float3 surface = preview_shade(matI, texI, texF, images, perlinVectors, perlinPermutations, &hit, direction);
                if (matI[hit.material * 4] != 2 || layer == PREVIEW_LAYERS - 1) {
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
                if (GUIDES_ON) normalSum += first.normal;
#if GUIDES_ON
                {
                    // How far this point moved across the screen since the frame before: where it
                    // is seen now, less where it was seen then - by the camera as it stood, and,
                    // on a mesh a turntable turns (OptiX knows which), where the mesh stood.
                    float3 p = first.point;
                    float3 before = p;
#if RT_OPTIX
                    if (first.turning) {
                        float dx = p.x - params.turnAxisX;
                        float dz = p.z - params.turnAxisZ;
                        before = make_float3(params.turnAxisX + params.turnCos * dx + params.turnSin * dz, p.y,
                                             params.turnAxisZ - params.turnSin * dx + params.turnCos * dz);
                    }
#endif
                    float2 now = screen_position(p, cameraCenter, pixel00, pixelDeltaU, pixelDeltaV);
                    float2 then = screen_position(before, previousCenter, previousPixel00, previousDeltaU, previousDeltaV);
                    flowX += now.x - then.x;
                    flowY += now.y - then.y;
                }
#endif
            }
#if GUIDES_ON
            else {
                // The background, far along the ray: it moves only as the camera turns.
                float3 far = origin + direction * (1.0e4f / length(direction));
                float2 now = screen_position(far, cameraCenter, pixel00, pixelDeltaU, pixelDeltaV);
                float2 then = screen_position(far, previousCenter, previousPixel00, previousDeltaU, previousDeltaV);
                flowX += now.x - then.x;
                flowY += now.y - then.y;
            }
#endif
            pixelColor += guide;
            sample++;
            fresh = true;
            continue;
        }

        bool done = false;
        HitRecord rec;
        if (world_hit(&scene, rootNode, origin, direction, time, 0.001f, INFINITY, &rng, &rec)) {
#if CAUSTICS
            bool photonsLight = false;
            if (causticState < 3) {
                int kind = matI[rec.material * 4];
                bool caster = is_caster(matI, matF, texI, texF, images, perlinVectors, perlinPermutations, &rec);
                if (causticState == 0) {
                    if (!caster) {
                        if (kind == 0) {
                            float3 albedo = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                                          matI[rec.material * 4 + 1], rec.u, rec.v, rec.point);
                            causticColor += throughput * albedo * (1.0f / M_PI_F) * caustic_gather(caustic, rec.point, rec.normal);
                            causticState = 1;
                        }
#if HAS_MEDIA
                        else if (kind == 4) {
                            float3 albedo = texture_value(texI, texF, images, perlinVectors, perlinPermutations,
                                                          matI[rec.material * 4 + 1], rec.u, rec.v, rec.point);
                            causticColor += throughput * albedo * (1.0f / (4.0f * M_PI_F)) * caustic_gather_volume(caustic, rec.point);
                            causticState = 1;
                        }
#endif
                        else {
                            causticState = 3;
                        }
                    }
                } else if (kind == 3 && causticState == 2) {
                    photonsLight = true;
                } else {
                    causticState = caster ? 2 : 3;
                }
            }
            if (!photonsLight)
#endif
            sampleColor += throughput * material_emitted(matI, matF, texI, texF, images,
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
                } else if (matI[rec.material * 4] != 4 && dot(chosen, rec.gnormal) <= 0.0f) {
                    // A direction under the surface itself - which a smooth normal can lean into -
                    // would leak light through it. (Fog scatters every way; it has no surface.)
                    done = true;
                } else {
                    // Weigh the sample by how much the surface actually scatters that way;
                    // that division is what keeps the estimate unbiased.
                    float surface = scattering_pdf(matI[rec.material * 4], rec.normal, chosen);
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
            float3 escaped = sky_colour(env, envWidth, envHeight, envScale, envTurn, direction, background, flatBackground);
            sampleColor += throughput * escaped;
            done = true;
        }

        depth++;
        if (done || depth >= maxReflectionDepth) {
            // A firefly - a rare path far brighter than its share - is capped, not averaged in raw.
            if (fireflyClamp > 0.0f) {
                float brightest = fmaxf(sampleColor.x, fmaxf(sampleColor.y, sampleColor.z));
                if (brightest > fireflyClamp) sampleColor *= fireflyClamp / brightest;
            }
            // One bad sample (NaN) would spoil the pixel for good; it is left out instead.
            if (sampleColor.x == sampleColor.x && sampleColor.y == sampleColor.y && sampleColor.z == sampleColor.z) {
                pixelColor += sampleColor;
            }
#if CAUSTICS
            if (causticColor.x == causticColor.x && causticColor.y == causticColor.y && causticColor.z == causticColor.z) {
                pixelColor += causticColor;
            }
#endif
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
    if (GUIDES_ON) {
        normalSum *= pixelSamplesScale;
        if (sampleStart > 0) normalSum += make_float3(guideNormal[base], guideNormal[base + 1], guideNormal[base + 2]);
        guideNormal[base] = normalSum.x; guideNormal[base + 1] = normalSum.y; guideNormal[base + 2] = normalSum.z;
        // The flow, averaged like the rest: (x, y) in pixels of this image, the third float unused.
        flowX *= pixelSamplesScale;
        flowY *= pixelSamplesScale;
        if (sampleStart > 0) { flowX += guideFlow[base]; flowY += guideFlow[base + 1]; }
        guideFlow[base] = flowX; guideFlow[base + 1] = flowY; guideFlow[base + 2] = 0.0f;
    }
}
