#include "pmpp__convolution.h"



__global__
void pmpp__2d_convolution_kernel(double *input, double *output, const int width,  const int height, const double* __restrict__ mask, const int mask_width){
    extern __shared__ double shared[];
    const int O_TILE_WIDTH = blockDim.x; //The output tile width is equal to the blockDim.x

    const int SHARED_STRIDE = blockDim.x + mask_width - 1; //The stride to the linearized shared memory array

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //The output array coordinates
    int row_o = blockIdx.y*O_TILE_WIDTH + ty;
    int col_o = blockIdx.x*O_TILE_WIDTH + tx;

    //The input array coordinates
    int row_i = row_o - mask_width/2;
    int col_i = col_o - mask_width/2;

    //Loads the input values or zero(to the ghosts and halo cells)
    if(row_i >= 0 && row_i < height && col_i >=0 && col_i < width){
        shared[ty*SHARED_STRIDE + tx] = input[row_i * width + col_i];
    }else{
        shared[ty*SHARED_STRIDE + tx] = 0.0;
    }

    __syncthreads();

    double value = 0.0;


    if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){ //Check if the thread index is into the output tile width
        for(int i = 0; i < mask_width; i++){ // Gets the rows and cols of the mask
          for(int j = 0; j < mask_width; j++){
              value += mask[i*mask_width + j] * shared[(i+ty)*SHARED_STRIDE + (j+tx)]; // Performs the convolution
          }
        }
        if(row_o < height && col_o < width){ //Check if thread indexes are into the output domain
            output[row_o*width + col_o] = value;
        }
    }
}

void pmpp__2d_convolution_host(double *input, double *output, const int width, const int height, const double *mask, const int mask_width){
    int ghosts_by_side = mask_width/2;
    double sum;
    int input_row, input_col;

    for(int output_row = 0; output_row < height; output_row++){ // Iterates through each output position (row and col) to calculate it
      for(int output_col = 0; output_col < width; output_col++){
          sum = 0;
          for(int mask_row = 0; mask_row < mask_width; mask_row++){   // Iterates through each mask position (row) to get it
            input_row = output_row - ghosts_by_side + mask_row;   //Calculates the input row index to be accessed
            if(input_row >= 0 && input_row < height ){           //Checks if the row has input values or only ghosts
              for(int mask_col = 0; mask_col < mask_width; mask_col++){  // Iterates through each mask position (col) to get it
                  input_col = output_col - ghosts_by_side + mask_col;  //Calculates the input col index to be accessed
                  if(input_col >= 0  && input_col < width){  //Checks if the col of the current row is a input value or a ghost cell
                      sum += input[input_row*width + input_col]*mask[mask_row*mask_width + mask_col]; //Performs the convolution
                  }
              }
              output[output_row*width + output_col] = sum;
            }
          }
      }
    }
}

void pmpp__1d_convolution_host(double *input, double *output, const int length, const double *mask, const int mask_width){
	int ghosts_by_side = mask_width/2;
	double sum;
	int input_idx;

	for(int out_idx = 0; out_idx < length; out_idx++){ // Iterates through each output position to calculate it
		sum = 0;
		for(int mask_idx = 0; mask_idx < mask_width; mask_idx++){ // Iterates through each mask position
			input_idx = out_idx - ghosts_by_side + mask_idx; // Calculates the input index
			if(input_idx >= 0 && input_idx < length){ // Check if the input index is not a ghost
				sum+=input[input_idx]*mask[mask_idx]; //Performs the convolution
			}
		}
		output[out_idx] = sum;
	}
}

__global__
void pmpp__1d_convolution_kernel(double *input, double *output, const int length, double *mask, const int mask_width){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double shared[];

	// Each thread loads data from global to the block shared memory
	shared[threadIdx.x] = tid < length ? input[tid] : 0.0;
	__syncthreads();

	// Defines the data index that belongs to each tile
	int this_tile_start_point = blockIdx.x * blockDim.x;
	int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;

	// Go back int(mask_width/2) positions in order to start from the block scope external cells (halos or ghosts placed before the this_tile_start_point position)
	int n_start_point = tid - (mask_width/2);
	double p = 0;

	for(int j = 0; j < mask_width; j++){

		int n_index = n_start_point + j;
		if(n_index >= 0 && n_index < length){ //Check if the n_index not refers to a ghost cell
			if(n_index >= this_tile_start_point && n_index < next_tile_start_point){ // If is an internal cell (true) or a halo cell (false)
				p += shared[threadIdx.x + j - mask_width/2]*mask[j];
			}else{
				p += input[n_index] * mask[j]; //Takes the N[value] from the cache (Luckily!) or from the global memory and performs the convolution
			}
		}
	}
	output[tid] = p;
}
