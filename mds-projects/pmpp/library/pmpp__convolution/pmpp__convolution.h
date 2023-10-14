#ifndef PMPP__CONVOLUTION_H_
#define PMPP__CONVOLUTION_H_


/**
* Kernel that performs 1D convolution operation
* @param input The input array
* @param output The output array
* @param length The input/output array length
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
__global__ void pmpp__1d_convolution_kernel(double *input, double *output, const int length, double *mask, const int mask_width);

/**
* The host function that performs 1D convolution operation
* @param input The input array
* @param output The output array
* @param length The input/output array length
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
void pmpp__1d_convolution_host(double *input, double *output, const int length, const double *mask, const int mask_width);


/**
* Kernel that performs 2D convolution operation
* @param input The input array
* @param output The output array
* @param width The input/output array width
* @param height The input/output array width
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
__global__ void pmpp__2d_convolution_kernel(double *input, double *output, const int width,  const int height, const double *mask, const int mask_width);


/**
* The host function that performs 2D convolution operation
* @param input The input array
* @param output The output array
* @param width The input/output array width
* @param height The input/output array height
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
void pmpp__2d_convolution_host(double *input, double *output, const int width, const int height, const double* __restrict__ mask, const int mask_width);

#endif /* PMPP__CONVOLUTION_H_ */
