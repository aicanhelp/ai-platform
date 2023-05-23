CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

project(mk-cuda-exmaples)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#include_directories(include)
#outpath
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

#Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_libraries(${OpenCV_LIBS})
#set(LIBS ${${OpenCV_LIBS}})

if (NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
else()
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif (NOT OpenCV_FOUND)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

#CUDA
find_package(CUDA)
if (CUDA_FOUND)
    FILE(GLOB DIRS Chapter*)

    foreach(DIR ${DIRS})
        get_filename_component(DIR_NAME ${DIR} NAME)
        FILE(GLOB_RECURSE SRCS ${DIR}/*.cu)
        foreach (SRC ${SRCS})
            get_filename_component(LIB_NAME ${SRC} NAME_WLE)
            CUDA_ADD_EXECUTABLE(${DIR_NAME}_${LIB_NAME} ${SRC})
            SET_PROPERTY(TARGET ${DIR_NAME}_${LIB_NAME} PROPERTY FOLDER ${DIR_NAME}_${LIB_NAME})
            target_link_libraries(${DIR_NAME}_${LIB_NAME} ${LIBS})
        endforeach ()

        FILE(GLOB_RECURSE CPP_SRCS ${DIR}/*.cpp)
        foreach (CPP_SRC ${CPP_SRCS})
            get_filename_component(CPP_LIB_NAME ${CPP_SRC} NAME_WLE)
            add_executable(${DIR_NAME}_${CPP_LIB_NAME}_cpp ${CPP_SRC})
            SET_PROPERTY(TARGET ${DIR_NAME}_${CPP_LIB_NAME}_cpp PROPERTY FOLDER ${DIR_NAME}_${CPP_LIB_NAME}_cpp)
#            target_link_libraries(${DIR_NAME}_${CPP_LIB_NAME}_cpp ${LIBS})
        endforeach ()
    endforeach()
else ()
    message("CUDA not found!")
endif ()
