#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */

/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles block (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304 matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // DONE: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // COMMENT: we can unroll j to allow for ILP
    // COMMENT: memory access to ouput is not coallesced (stride = n, not 1).
    //          it touches 32 cache lines (assuming n > 32).
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // DONE: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // each block takes care of a chunk of size 64 x 64
    __shared__ float data[64][65]; // 64 x 64 block with an extra column of padding

    // blockDim.x = 4*blockDim.y = 64
    int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    int end_j = j + 4;
    
    // copy input to shared memory
    const int li = threadIdx.x;
    int lj = 4*threadIdx.y;
    for (; j < end_j; j++, lj++) {
        // we're using a stride of 1 for global memory --> memory coallescing
        data[lj][li] = input[i + j * n]; 
        // note: even without the padding, no bank conflicts would occur here
        // elements in data[lj][:] belong to different banks
    }

    __syncthreads(); // sync with all other threads in current block

    // reset some indices
    i = threadIdx.x + 64 * blockIdx.y;
    j = 4 * threadIdx.y + 64 * blockIdx.x;
    lj = 4*threadIdx.y;
    end_j = j + 4;
    // copy output to global memory
    for (; j < end_j; j++, lj++) {
        // we're using a stride of 1 for global memory --> memory coallescing
        output[i + j * n] = data[li][lj];
        // the padding is necesseray to prevent bank conflicts here
        // each element in data[:][lj] belongs to different banks
    }
    // COMMENT: for loops should be unrolled for ILP
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.

__shared__ float data[64][65]; // 64 x 64 block with an extra column of padding

    // blockDim.x = 4*blockDim.y = 64
    int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    
    // copy input to shared memory
    const int li = threadIdx.x;
    const int lj = 4*threadIdx.y;
    // we've manually unrolled the for loop
    data[lj    ][li] = input[i +  j      * n]; 
    data[lj + 1][li] = input[i + (j + 1) * n]; 
    data[lj + 2][li] = input[i + (j + 2) * n]; 
    data[lj + 3][li] = input[i + (j + 3) * n]; 

    __syncthreads(); // sync with all other threads in current block

    // reset some indices
    i = threadIdx.x + 64 * blockIdx.y;
    j = 4 * threadIdx.y + 64 * blockIdx.x;

    // copy output to global memory
    // we've manually unrolled the for loop
    output[i + j       * n] = data[li][lj    ];
    output[i + (j + 1) * n] = data[li][lj + 1];
    output[i + (j + 2) * n] = data[li][lj + 2];
    output[i + (j + 3) * n] = data[li][lj + 3];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}