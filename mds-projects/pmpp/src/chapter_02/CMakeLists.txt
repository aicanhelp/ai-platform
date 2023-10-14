CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(vec_add ch2__vec_add.cu)
SET_PROPERTY(TARGET vec_add  PROPERTY FOLDER chapter_02)
target_link_libraries(vec_add ${LIBS})
