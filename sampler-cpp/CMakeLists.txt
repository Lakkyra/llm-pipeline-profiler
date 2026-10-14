cmake_minimum_required(VERSION 3.10)
project(sampler_cpp CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA compiler (nvcc)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA OPTIONAL)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}. Building hardware_prober with CUDA.")
else()
    message(STATUS "CUDA compiler not found. Building hardware_prober with C++ fallback.")
    set_source_files_properties(src/hardware_prober.cu PROPERTIES LANGUAGE CXX)
endif()

# Add source files
set(SOURCES
    src/proc_reader.cpp
    src/nvml_reader.cpp
    src/ipc_server.cpp
    src/hardware_prober.cu
    src/main.cpp
    src/standalone_main.cpp
)

# Header search paths
include_directories(src)

# Create standalone executable target
add_executable(llm_profiler_sampler_bin ${SOURCES})

# Windows and Linux specific configurations
if(WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(llm_profiler_sampler_bin PRIVATE Threads::Threads)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(llm_profiler_sampler_bin PRIVATE dl Threads::Threads)
endif()

# Set output name to something descriptive
set_target_properties(llm_profiler_sampler_bin PROPERTIES OUTPUT_NAME llm_profiler_sampler)
