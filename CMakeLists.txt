cmake_minimum_required(VERSION 3.10)
project(libml VERSION 0.0.1)

# Set Compiler
set(CMAKE_C_COMPILER gcc)

# Set C standard to C23 (manually setting compile flags)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c2x")

# Include directory with header files
include_directories(include)

# Collect source and header files
file(GLOB SOURCES "src/*.c")
file(GLOB HEADERS "include/*.h")

# Compile the library
add_library(ml SHARED ${SOURCES})

# Install the library and headers
install(TARGETS ml LIBRARY DESTINATION /usr/local/lib)
install(FILES ${HEADERS} DESTINATION /usr/local/include/libml)
