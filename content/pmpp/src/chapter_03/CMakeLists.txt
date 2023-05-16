CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

CUDA_ADD_EXECUTABLE(blur ch3__blur.cu)
SET_PROPERTY(TARGET blur  PROPERTY FOLDER chapter_03)
target_link_libraries(blur ${LIBS})

CUDA_ADD_EXECUTABLE(color_to_grayscale ch3__color_to_grayscale.cu)
SET_PROPERTY(TARGET color_to_grayscale  PROPERTY FOLDER chapter_03)
target_link_libraries(color_to_grayscale ${LIBS})