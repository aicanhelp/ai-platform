cmake_minimum_required(VERSION 3.10)
project(flink_statebackend VERSION 0.1.0 LANGUAGES C CXX)

# 添加头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/ubs_mem/include
    )

# 指定源文件
file(GLOB SRC_FILES "src/*.cpp")

link_directories(/usr/local/ubs_mem/lib)

# 创建可执行文件
add_executable(UBStateBackend ${SRC_FILES})
target_link_libraries(UBStateBackend PUBLIC ubsm_sdk)