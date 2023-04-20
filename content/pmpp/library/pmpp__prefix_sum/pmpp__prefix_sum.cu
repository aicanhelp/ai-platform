#include <math.h>
#include "pmpp__prefix_sum.h"


__global__
void pmpp__single_pass_kogge_stone_full_scan_kernel(double *input, double *output, const int length, volatile double *scan_value, unsigned int *flags, unsigned int *block_counter){
	extern __shared__ double spks_section_sums[];
	__shared__ float previous_sum;
	__shared__ int sbid;

	if(threadIdx.x == 0){
		sbid = atomicAdd(block_counter, 1);
	}

	__syncthreads();

	const int bid = sbid;
	const int tid = bid*blockDim.x + threadIdx.x;


	if(tid < length){
		spks_section_sums[threadIdx.x] = input[tid];
	}else{
		spks_section_sums[threadIdx.x] = 0.0;
	}

	unsigned int stride;
	for( stride= 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		if(threadIdx.x >= stride){
			spks_section_sums[threadIdx.x] += spks_section_sums[threadIdx.x - stride];
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		while(atomicAdd(&flags[bid], 0) == 0){;};
		previous_sum = scan_value[bid];
		scan_value[bid + 1]  = previous_sum + spks_section_sums[blockDim.x -1];
		__threadfence();
		atomicAdd(&flags[bid + 1], 1);
	}
	__syncthreads();

	if(tid < length){
		output[tid] = previous_sum + spks_section_sums[threadIdx.x];
	}
}

__global__
void pmpp__kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double ks_section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	if(tid < length){
		ks_section_sums[threadIdx.x] = input[tid];
	}else{
		ks_section_sums[threadIdx.x] = 0.0;
	}

	__syncthreads();

	unsigned int stride;
	for( stride= 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		if(threadIdx.x >= stride){
			ks_section_sums[threadIdx.x] += ks_section_sums[threadIdx.x - stride];
		}
	}

	__syncthreads();

	if(tid < length){
		output[tid] = ks_section_sums[threadIdx.x];
	}

	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = ks_section_sums[threadIdx.x];
	}
}

__global__
void pmpp__brent_kung_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double bk_section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(tid < length){
		bk_section_sums[threadIdx.x] = input[tid];
	}else{
		bk_section_sums[threadIdx.x] = 0.0;
	}

	__syncthreads();


	for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 * stride - 1;
		if(idx < blockDim.x && (idx - stride) < blockDim.x){
			bk_section_sums[idx] += bk_section_sums[idx - stride];
		}
	}

	for(int stride = blockDim.x/2; stride > 0; stride /=2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 *stride - 1;
		if((idx + stride) < blockDim.x && idx < blockDim.x){
			bk_section_sums[idx + stride] += bk_section_sums[idx];
		}
	}
	__syncthreads();

	if(tid < length){
		output[tid] = bk_section_sums[threadIdx.x];
	}

	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = bk_section_sums[threadIdx.x];
	}
}

__global__
void pmpp__3_phase_kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, const int section_length, double *last_sum){
	extern __shared__ double ks3p_section_sums[];
	int b_dim = blockDim.x;

	// How many phases we should have in order to load the input array to shared memory in a coalesced manner (corner turning)
	int phases_count = ceil(section_length/(double)b_dim);
	// The subsection length is setted to be equals to the phases_count, in order to use all threads in the subsection scan
	int sub_section_max_length = phases_count;


	// Phase 1: Corner turning to load the input data into shared memory
	for(int i = 0; i < phases_count; i++){
		int shared_mem_index = i*b_dim + threadIdx.x;
		int input_index = blockIdx.x*section_length + shared_mem_index;
		//This comparison could be removed if we handle the last phase separately and using the dynamic blockIndex assignment
		if(shared_mem_index < section_length){
			if(input_index < length){
				ks3p_section_sums[shared_mem_index] = input[input_index];
			}else{
				ks3p_section_sums[shared_mem_index] = 0.0;
			}

		}
	}

	__syncthreads();

	//Phase 1: Perform the scan on each sub_section
	for(int i = 1; i < sub_section_max_length; i++){
		int index = threadIdx.x*sub_section_max_length + i;
		if(index < section_length){
			ks3p_section_sums[index] += ks3p_section_sums[index -1];
		}
	}

	__syncthreads();


	//Phase 2: Performs the Kogge-Stone scan for the last element of each subsection. This step could be performed also by Brent-Kung scan
	for(int stride= 1; stride < section_length; stride *= 2){
		__syncthreads();
		// sub_section_length*threadIdx.x: Indicates the start position of each subsection
		// sub_section_length -1: The last item in a given subsection
		int last_element = sub_section_max_length*threadIdx.x + sub_section_max_length -1;
		if(threadIdx.x >= stride && last_element < section_length){
			ks3p_section_sums[last_element] += ks3p_section_sums[last_element - stride*sub_section_max_length];
		}
	}




	__syncthreads();

	//Phase 3: Adding the last element of previous sub_section
	for(int i = 0; i < sub_section_max_length - 1; i++){
		__syncthreads();
		if(threadIdx.x != 0){
			int index = threadIdx.x*sub_section_max_length + i;
			if(index < section_length){
				ks3p_section_sums[index] += ks3p_section_sums[threadIdx.x*sub_section_max_length - 1];
			}
		}
	}

	//Save the data on the output array
	for(int i = 0; i < phases_count; i++){
		int output_index = blockIdx.x*section_length + i*b_dim + threadIdx.x;
		if(i*b_dim + threadIdx.x < section_length){
			output[output_index] = ks3p_section_sums[i*b_dim + threadIdx.x];
		}
	}

	if(last_sum != NULL && threadIdx.x == 0){
		last_sum[blockIdx.x] = ks3p_section_sums[section_length - 1];
	}


}

