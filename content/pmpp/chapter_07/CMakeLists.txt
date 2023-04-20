CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(1d_convoluton ch7__1d_convolution.cu)
SET_PROPERTY(TARGET 1d_convoluton  PROPERTY FOLDER chapter_07)

CUDA_ADD_EXECUTABLE(2d_convoluton ch7__2d_convolution.cu)
SET_PROPERTY(TARGET 2d_convoluton  PROPERTY FOLDER chapter_07)