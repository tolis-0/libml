cmake_minimum_required(VERSION 3.10)
project(libml VERSION 0.0.1)

# Set Compiler to gcc and C version to C23
set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c2x")

# Conditionally add compiler flags for debugging and general use
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug Compilation")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Wextra")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=undefined")
else()
    message(STATUS "Standard Compilation")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native -flto")
endif()

# Include directory with header files
include_directories(include)

# Collect source and header files
file(GLOB_RECURSE SOURCES "src/*.c")
file(GLOB HEADERS "include/*.h")

# Compile the library
add_library(ml SHARED ${SOURCES})

# Install the library and headers
install(TARGETS ml LIBRARY DESTINATION /usr/local/lib)
install(FILES ${HEADERS} DESTINATION /usr/local/include/ml)
