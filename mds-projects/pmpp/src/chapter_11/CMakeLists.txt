CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(merge_sort ch11__merge_sort.cu)
SET_PROPERTY(TARGET merge_sort  PROPERTY FOLDER chapter_11)
