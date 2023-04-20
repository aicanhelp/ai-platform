#include "pmpp__reduction.h"

__global__
void pmpp__sum_by_block_interleaved_kernel(double *v, double *sum, const int length){
    extern __shared__ double partial_sum[];
    unsigned int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;

    // Copies the elements to be added to the shared memory. If the value is a garbage one, includes zero instead
    partial_sum[tx] = tid < length ? v[tid] : 0.0;

    // Halve the stride in each iteration, bringing the temporary sums into the first half
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(tx % (2*stride) == 0){ // Check if thread is inthe first half
            partial_sum[tx] += partial_sum[tx+stride];
        }
    }
    __syncthreads();
    sum[blockIdx.x] = partial_sum[0];    
}

__global__
void pmpp__sum_by_block_kernel(double *v, double *sum, const int length){
    extern __shared__ double partial_sum[];
    unsigned int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;

    // Copies the elements to be added to the shared memory. If the value is a garbage one, includes zero instead
    partial_sum[tx] = tid < length ? v[tid] : 0.0;

    // Halve the stride in each iteration, bringing the temporary sums into the first half
    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(tx < stride){ // Check if thread is inthe first half
            partial_sum[tx] += partial_sum[tx+stride];
        }
    }
    __syncthreads();
    sum[blockIdx.x] = partial_sum[0];    
}


double pmpp__sum_host(double *v, const int length){
  double temp = 0.0;
  for(int i = 0; i < length; i++){
      temp+= v[i];
  }

  return temp;
}
