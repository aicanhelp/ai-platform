CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(parallel_histogram ch9__parallel_histogram.cu)
SET_PROPERTY(TARGET parallel_histogram  PROPERTY FOLDER chapter_09)
