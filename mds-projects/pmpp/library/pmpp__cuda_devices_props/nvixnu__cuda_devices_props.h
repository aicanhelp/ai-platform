#ifndef NVIXNU__CUDA_DEVICES_PROPS_H_
#define NVIXNU__CUDA_DEVICES_PROPS_H_

/**
* Prints the main properties of a CUDA device.
* These properties are useful for programmers to allocate resources and manage the occupancy
*/
void nvixnu__print_cuda_devices_props(void);

/**
* Returns the dev_props object populated by the cudaGetDeviceProperties function
* @param device_number The device number from which the properties are requested
* @return A cudaDeviceProp object
*/
cudaDeviceProp nvixnu__get_cuda_device_props(int device_number);

#endif /* NVIXNU__CUDA_DEVICES_PROPS_H_ */