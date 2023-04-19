/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 5
 * In this chapter the partial vector sum (reduction) is presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 06/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_5_H_
#define CHAPTER_5_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH5__FILEPATH CREDIT_CARD_DATASET_PATH
#define CH5__ARRAY_LENGTH (CREDIT_CARD_DATASET_LENGTH - 1) //In this dataset the sum reduction is equals to zero. Subtracting 1, the sum should be equals to the negative last number (0.013649)

/**
 * Performs the host and device array sum (reduction)
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch5__sum_reduction(env_e env, kernel_config_t config);


#endif /* CHAPTER_5_H_ */
