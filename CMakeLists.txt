cmake_minimum_required(VERSION 3.18)
project(test_task_dl_26_12_2025)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
    ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/vendor/onnxruntime-linux-x64-gpu-1.23.2"
)
set(ARGPARSE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendor/argparse-3.2/include")

find_package(OpenCV REQUIRED)

include_directories(
    ${ONNXRUNTIME_ROOT}/include 
    ${ARGPARSE_INCLUDE_DIR}
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(demo_detection src/cpp/main.cpp)
target_link_libraries(demo_detection ${OpenCV_LIBS} onnxruntime)

add_custom_command(TARGET demo_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    $<TARGET_FILE_DIR:demo_detection>)
