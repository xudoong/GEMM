cmake_minimum_required(VERSION 3.19)
project(NVIDIA_HSGEMM LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 80)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/temp)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20054 --keep --keep-dir ${CMAKE_BINARY_DIR}/temp")

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
# add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/dssg/home/acct-hpc/hpcwxd/repos/cutlass/include)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC1)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/kernels SRC2)

# generate executable
add_executable(hsgemm main.cu ${SRC1} ${SRC2})
set_target_properties(hsgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(hsgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

target_compile_options(hsgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       >)