# DepthEstimation-SGM

# 1. Prerequisites
We have tested the library in **Ubuntu 16**, but it should be easy to compile in other platforms.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Sophus
This is an Lie algebra library. More details can be found in https://github.com/strasdat/Sophus.
It's worth mentioning that we adopted an old version of the library, so please use "git reset" to convert the head to to "a621ff2".

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **We use 3.4.1, but it should also work for other version at least 3.0**.

## Eigen3
Download and install instructions can be found at: http://eigen.tuxfamily.org.


## Cuda
We complie the project with Cuda 10, theoretically compiling with other versions are also feasible.

# 2. Build and run the project
We use CMake to build the project on ubuntu 16.
```
cd DepthEstimation-SGM/
mkdir build
cd build
cmake ..
make
cd ..
./build/main_covins
```
