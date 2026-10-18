# VaaniEdge custom ARM64 NEON DSP library.
# Builds libvaani_dsp_neon.so for arm64-v8a only (Snapdragon 8+ Gen 1 target).
cmake_minimum_required(VERSION 3.22.1)
project(vaani_dsp_neon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vaani_dsp_neon SHARED vaani_dsp_neon.cpp)

# AArch64 has mandatory NEON; -O3/-ffast-math for the hot Mel/FFT path.
target_compile_options(vaani_dsp_neon PRIVATE
        -O3
        -ffast-math
        -funroll-loops
        -Wall)

find_library(log-lib log)
target_link_libraries(vaani_dsp_neon ${log-lib})
