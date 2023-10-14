#ifndef PMPP__PREFIX_SUM_H_
#define PMPP__PREFIX_SUM_H_

/**
 * Performs the core of sequential prefix sum
 * @param input The input array
 * @param output The output array
 * @param length The arrays length
 * @param stride The stride (distance between an element to another). 1 for regular arrays and > 1 for "virtual" arrays
 */
__host__
__device__
__attribute__((always_inline))
static inline void pmpp__partial_prefix_sum_unit(double *input, double *output, const int length, const int stride){
	double acc = input[stride - 1];
	output[0] = acc;
	for(int i = stride; i < length; i=i+stride){
		acc += input[i];
		output[i] = acc;
	}
}

/**
 * Performs the single pass Kogge-Stone prefix sum on arbitrary sized inputs.
 * @param input The input array
 * @param output The output array
 * @param length The arrays length
 * @param scan_value Auxiliary array that holds the sum of each block
 * @param flags Auxiliary array used to orchestrate the adjacent block synchonization.
 * @param block_counter Auxiliary variable that holds the dynamic block id used in the adjacent block synchronization.
 */
__global__
void pmpp__single_pass_kogge_stone_full_scan_kernel(double *input, double *output, const int length, volatile double *scan_value, unsigned int *flags, unsigned int *block_counter);


/**
 * Performs the Kogge-Stone prefix sum by block. The length of each "partial scan" section is up to blockDim.x.
 * @param input The input array
 * @param output The output array
 * @param length The arrays length
 * @param last_sum Global memory array pointer (For hierarchical scan) or NULL. The global array holds the scan value of the last section's element.
 */
__global__
void pmpp__kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum);


/**
 * Performs the Brent-Kung prefix sum by block. The length of each "partial scan" section is up to blockDim.x.
 * @param input The input array
 * @param output The output array
 * @param length The arrays length
 * @param last_sum Global memory array pointer (For hierarchical scan) or NULL. The global array holds the scan value of the last section's element.
 */
__global__
void pmpp__brent_kung_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum);

/**
 * Performs the 3-phase Kogge-Stone prefix sum by block. The length of each "partial scan" section is up to sharedMemPerBlock.
 * @param input The input array
 * @param output The output array
 * @param length The arrays length
 * @paramsection_length The length of each section, that is, the length of shared memory
 * @param last_sum Global memory array pointer (For hierarchical scan) or NULL. The global array holds the scan value of the last section's element.
 */
__global__
void pmpp__3_phase_kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, const int section_length, double *last_sum);


#endif /* PMPP__PREFIX_SUM_H_ */
