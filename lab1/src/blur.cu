__host__ __device__ void cuda_blur_kernel_convolution(uint thread_index, 
                                  const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.
    if (thread_index < blur_v_size) {
        for (int j = 0; j <= thread_index; j++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - j]*gpu_blur_v[j];
        }
    }
    else {
        for (int j = 0; j < blur_v_size; j++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - j]*gpu_blur_v[j];
        }
    }
}

__global__ void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // DONE: Compute the current thread index. (1D grid and block)
    uint thread_index = blockDim.x*blockIdx.x + threadIdx.x; 

    // DONE: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    while (thread_index < n_frames) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        // DONE: Update the thread index
        thread_index += blockDim.x*gridDim.x;
    }
}

// This function will be called from the host code to invoke the kernel
// function. Any memory address/pointer locations passed to this function
// must be host addresses. This function will be in charge of allocating
// GPU memory, invoking the kernel, and cleaning up afterwards. The result
// will be stored in out_data. The function returns the amount of time that
// it took for the function to complete (prior to returning) in milliseconds.
float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    // DONE: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data. The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.
    float* gpu_raw_data;
    cudaMalloc((void**) &gpu_raw_data, n_frames*sizeof(float));
    cudaMemcpy(gpu_raw_data, raw_data, n_frames*sizeof(float), cudaMemcpyHostToDevice);

    // DONE: Allocate GPU memory for the impulse signal (for now global GPU
    //       memory is fine. The data is of type float and has blur_v_size
    //       elements. Then copy the data in blur_v into the GPU memory you
    //       allocated.
    float* gpu_blur_v;
    cudaMalloc((void**) &gpu_blur_v, blur_v_size*sizeof(float));
    cudaMemcpy(gpu_blur_v, blur_v, blur_v_size*sizeof(float), cudaMemcpyHostToDevice);

    // Done: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    cudaMalloc((void**) &gpu_out_data, n_frames*sizeof(float));
    cudaMemset(gpu_out_data, 0, n_frames*sizeof(float));

    // DONE: Appropriately call the kernel function
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    // DONE: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)
    cudaMemcpy(out_data, gpu_out_data, n_frames*sizeof(float), cudaMemcpyDeviceToHost);

    // DONE: Free GPU resources
    cudaFree(gpu_raw_data);
    cudaFree(gpu_blur_v);
    cudaFree(gpu_out_data);
    
    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}