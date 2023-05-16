CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(full_prefix_sum ch8__full_prefix_sum.cu)
SET_PROPERTY(TARGET full_prefix_sum  PROPERTY FOLDER chapter_08)

CUDA_ADD_EXECUTABLE(partial_prefix_sum ch8__partial_prefix_sum.cu)
SET_PROPERTY(TARGET partial_prefix_sum  PROPERTY FOLDER chapter_08)