CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

project(pmpp)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include_directories(include)
#outpath
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

#Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)


#CUDA
find_package(CUDA)
if (CUDA_FOUND)
    FILE(GLOB DIRS chapter_*)

    foreach (DIR ${DIRS})
        get_filename_component(DIR_NAME ${DIR} NAME)
        #    message("Tutorial: ${TUTORIAL_NAME}")
        add_subdirectory(${DIR_NAME})
    endforeach (DIRS)
else ()
    message("CUDA not found!")
endif ()
