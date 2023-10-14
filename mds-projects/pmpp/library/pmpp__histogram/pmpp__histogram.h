#ifndef PMPP__HISTOGRAM_H_
#define PMPP__HISTOGRAM_H_

/**
 * Performs the histogram on host
 */
void pmpp__histogram_host(char *input, const int length, int *output);

/**
 * Performs the histogram on device with naive/basic data partitioning
 */
__global__ void pmpp__histogram_with_block_partitioning_kernel(char *input, const int length, int *output);

/**
 * Performs the histogram on device with interleaved data partitioning
 */
__global__ void pmpp__histogram_with_interleaved_partitioning_kernel(char *input, const int length, int *output);

/**
 * Performs the histogram on device with data privatization
 */
__global__ void pmpp__histogram_privatized_kernel(char* input, const int input_length, int* output, const int output_length);

/**
 * Performs the histogram on device with data aggregation
 */
__global__ void pmpp__histogram_aggregated_kernel(char* input, const int input_length, int* output, const int output_length);


#endif /* PMPP__HISTOGRAM_H_ */
