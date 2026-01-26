cmake_minimum_required(VERSION 3.18)
project(test_task_dl_26_12_2025)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Paths for ONNX Runtime (Update to your actual local paths)
set(ONNXRUNTIME_ROOT "/path/to/onnxruntime-linux-x64-gpu-1.23.0")
# Path for p-ranav/argparse (If you just downloaded the header)
set(ARGPARSE_INCLUDE_DIR "/path/to/argparse/include")

include_directories(
    ${ONNXRUNTIME_ROOT}/include 
    ${ARGPARSE_INCLUDE_DIR}
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(demo_detection main.cpp)
target_link_libraries(demo_detection ${OpenCV_LIBS} onnxruntime)

# Post-build copy of the .so file for convenience
add_custom_command(TARGET demo_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    $<TARGET_FILE_DIR:demo_detection>)
