#include "pmpp__histogram.h"

__global__
void pmpp__histogram_aggregated_kernel(char* input, const int input_length, int* output, const int output_length) {
	extern __shared__ unsigned int agg_histo_s[];
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx +=blockDim.x) {
		agg_histo_s[binIdx] = 0;
	}
	__syncthreads();
	int prev_index = -1;
	int accumulator = 1;

	for(unsigned int i = tid; i < input_length; i += blockDim.x*gridDim.x) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			unsigned int curr_index = alphabet_position/4;
			if (curr_index != prev_index) {
				atomicAdd(&(agg_histo_s[alphabet_position/4]), accumulator);
				accumulator = 1;
				prev_index = curr_index;
			}
			else {
				accumulator++;
			}
		}
	}
	__syncthreads();

	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx += blockDim.x) {
		atomicAdd(&(output[binIdx]), agg_histo_s[binIdx]);
	}
}

__global__
void pmpp__histogram_privatized_kernel(char* input, const int input_length, int* output, const int output_length) {
	extern __shared__ unsigned int priv_histo_s[];

	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;


	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx +=blockDim.x) {
		priv_histo_s[binIdx] = 0;
	}
	__syncthreads();

	for (unsigned int i = tid; i < input_length; i += blockDim.x*gridDim.x) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26){
			atomicAdd(&(priv_histo_s[alphabet_position/4]), 1);
		}
	}
	__syncthreads();
	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx += blockDim.x) {
		atomicAdd(&(output[binIdx]), priv_histo_s[binIdx]);
	}
}

__global__
void pmpp__histogram_with_interleaved_partitioning_kernel(char *input, const int length, int *output){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (unsigned int i = tid; i < length; i += blockDim.x*gridDim.x ) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26){
			atomicAdd(&(output[alphabet_position/4]), 1);
		}
	}
}

__global__
void pmpp__histogram_with_block_partitioning_kernel(char *input, const int length, int *output){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int section_size = ceil(length/(double)(blockDim.x * gridDim.x));
	int start = i*section_size;

	for (int k = 0; k < section_size; k++) {
		if (start+k < length) {
			int alphabet_position = input[start+k] - 'a';
			if (alphabet_position >= 0 && alphabet_position < 26){
				atomicAdd(&(output[alphabet_position/4]), 1);
			}
		}
	}
}

void pmpp__histogram_host(char *input, const int length, int *output){
	for (int i = 0; i < length; i++) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			output[alphabet_position/4]++;
		}
	}
}
