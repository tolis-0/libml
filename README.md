# LibML
LibML is an efficient C library for machine learning designed with a simple user interface. Currently, it implements a flexible structure for feed-forward neural networks, with a lot more features in development.

## Requirements
- C11
- CMake 3.10 or higher
- CBLAS library

### Get CBLAS:
##### Ubuntu/Debian `sudo apt-get install libblas-dev`
##### Fedora `sudo dnf install blas-devel`
##### MacOS `brew install cblas`
Alternatively you can visit https://www.netlib.org/blas/ for the source code, or the OpenBLAS version at https://github.com/OpenMathLib/OpenBLAS which also includes Windows binaries.

## Installation
```sh
mkdir build
cd build
cmake ..
make
sudo make install
```
