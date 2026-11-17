// The photon map's grid: the photons a launch landed (raytracer.cu, AOV_MODE 5), sorted by the
// grid cell they fall in, so that a gather reads each cell's photons as one run. A counting sort:
// count per cell, an exclusive prefix sum of the counts, then each photon copied to its place.
// It knows nothing of the scene, so it is a module of its own, built once a session
// (gpu/CausticMap).

#define PHOTON_FLOATS 12
#define SCAN_BLOCK 1024

/** As raytracer.cu's caustic_cell: the two must hash alike. */
__device__ inline int caustic_cell(int x, int y, int z, int mask) {
    unsigned int h = ((unsigned int)x * 73856093u) ^ ((unsigned int)y * 19349663u) ^ ((unsigned int)z * 83492791u);
    return (int)(h & (unsigned int)mask);
}

/** Each landed photon's cell, into keys (-1 for an empty slot), counted into counts. */
extern "C" __global__ void caustic_count(const float* photons, int n, float cellSize, int mask, int* counts, int* keys) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float* p = photons + i * PHOTON_FLOATS;
    if (p[3] <= 0.0f) { keys[i] = -1; return; }
    int h = caustic_cell((int)floorf(p[0] / cellSize), (int)floorf(p[1] / cellSize), (int)floorf(p[2] / cellSize), mask);
    keys[i] = h;
    atomicAdd(&counts[h], 1);
}

/** The exclusive prefix sum of counts within each block of SCAN_BLOCK, and each block's total. */
extern "C" __global__ void caustic_scan_blocks(const int* counts, int* starts, int* blockSums, int n) {
    __shared__ int buffer[SCAN_BLOCK];
    int t = threadIdx.x;
    int i = (int)(blockIdx.x * SCAN_BLOCK) + t;
    int value = i < n ? counts[i] : 0;
    buffer[t] = value;
    __syncthreads();
    // Hillis-Steele, inclusive.
    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
        int add = t >= offset ? buffer[t - offset] : 0;
        __syncthreads();
        buffer[t] += add;
        __syncthreads();
    }
    if (i < n) starts[i] = buffer[t] - value;
    if (t == SCAN_BLOCK - 1) blockSums[blockIdx.x] = buffer[t];
}

/** The blocks' totals into where each block starts: one thread, a few thousand additions. */
extern "C" __global__ void caustic_scan_sums(int* blockSums, int blocks) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int running = 0;
    for (int b = 0; b < blocks; b++) {
        int total = blockSums[b];
        blockSums[b] = running;
        running += total;
    }
    blockSums[blocks] = running;
}

/** Each cell's start made global, a copy of it as the cursor photons are placed with, and the end. */
extern "C" __global__ void caustic_add_offsets(int* starts, const int* blockSums, int n, int blocks, int* cursor) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        int s = starts[i] + blockSums[i / SCAN_BLOCK];
        starts[i] = s;
        cursor[i] = s;
    }
    if (i == 0) starts[n] = blockSums[blocks];
}

/** Every landed photon copied to the next free place in its cell's run. */
extern "C" __global__ void caustic_scatter(const float* photons, const int* keys, int n, int* cursor, float* sorted) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int h = keys[i];
    if (h < 0) return;
    int at = atomicAdd(&cursor[h], 1);
    const float* from = photons + i * PHOTON_FLOATS;
    float* to = sorted + at * PHOTON_FLOATS;
    for (int k = 0; k < PHOTON_FLOATS; k++) to[k] = from[k];
}
