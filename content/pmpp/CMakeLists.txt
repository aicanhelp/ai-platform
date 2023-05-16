CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

project(pmpp)

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

#CUDA
find_package(CUDA)
if (CUDA_FOUND)
    FILE(GLOB LIB_DIRS library/*)

    foreach (LIB_DIR ${LIB_DIRS})
        get_filename_component(LIB_NAME ${LIB_DIR} NAME_WLE)
        file(GLOB FILES "${LIB_DIR}/*.*")
        CUDA_ADD_LIBRARY(${LIB_NAME} SHARED ${FILES})
        include_directories(${LIB_DIR})
        set(LIBS ${LIBS} ${LIB_NAME})
    endforeach ()
    message("${LIBS}")

    #    FILE(GLOB DIRS chapter_*)
    #
    #    foreach (DIR ${DIRS})
    #        get_filename_component(DIR_NAME ${DIR} NAME)
    #        #    message("Tutorial: ${TUTORIAL_NAME}")
    #        add_subdirectory(${DIR_NAME})
    #    endforeach (DIRS)
    FILE(GLOB_RECURSE SRCS src/*.cu)
    foreach (SRC ${SRCS})
        get_filename_component(LIB_NAME ${SRC} NAME_WLE)
        CUDA_ADD_EXECUTABLE(${LIB_NAME} ${SRC})
        SET_PROPERTY(TARGET ${LIB_NAME} PROPERTY FOLDER ${LIB_NAME})
        target_link_libraries(${LIB_NAME} ${LIBS})
    endforeach ()
else ()
    message("CUDA not found!")
endif ()
