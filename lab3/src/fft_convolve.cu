/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__device__ void PWMS(const cufftComplex a, const cufftComplex b, cufftComplex &c, const int N) {
    const int Ninv = 1./N;
    c.x = (a.x*b.x - a.y*b.y)*Ninv;
    c.y = (a.x*b.y + a.y*b.x)*Ninv;
}

__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    const int padded_length) {

    /* DONE: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    while (idx < padded_length) {
        PWMS(raw_data[idx], impulse_v[idx], out_data[idx], padded_length);
        idx += gridDim.x*blockDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE: Implement the maximum-finding (of the real parts).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    const int N_per_thread = floorf(padded_length/(gridDim.x * blockDim.x)) + 1;
    extern __shared__ float s_max[]; 

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // find the max out of all elements belonging to current thread
    // and store it in shared memory
    float max_idx = 0.0;
    while (idx < padded_length) {
        float mag = fabs(out_data[idx].x);
        if (max_idx < mag) {
            max_idx = mag;
        }
        idx += gridDim.x*blockDim.x;
    }

    // reset idx
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    s_max[idx] = max_idx;

    __syncthreads();

    int N = blockDim.x/2;
    while (idx < N) {
        if (s_max[threadIdx.x] < s_max[threadIdx.x + N]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + N];
        }

        N /= 2;
        __syncthreads();
    }

    // atomic max across all blocks
    if (threadIdx.x == 0) {
        atomicMax(max_abs_val, s_max[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < padded_length) {
        out_data[idx].x /= *max_abs_val;
        out_data[idx].y /= *max_abs_val;
        idx += gridDim.x *blockDim.x;
    }

}

void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* DONE: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* DONE: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, blockDim.x*sizeof(float)>>>(out_data, max_abs_val, padded_length);

}

void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* DONE: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}