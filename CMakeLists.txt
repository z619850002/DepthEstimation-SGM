#declare the minimum cmake version
cmake_minimum_required( VERSION 2.8 )


#This can be debug by the gdb
SET(CMAKE_BUILD_TYPE "Release") 
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -g -fopenmp") 
set(CUDA_USE_STATIC_CUDA_RUNTIME ON) 


#Support c++ 11
SET(CMAKE_CXX_FLAGS "-std=c++11")


#declare a cmake project
project(DepthEstimation)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


#use cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    #    -std=c++11 -O3 -gencode arch=compute_30,code=sm_30  # T440s
    #    -std=c++11 -O3 -gencode arch=compute_53,code=sm_53  # tx1
    -std=c++11 -O3 -gencode arch=compute_70,code=sm_70 # gtx 980 Ti / 1080
    )



#Eigen
include_directories( "/usr/include/eigen3" )

 
set(OpenCV_DIR "/usr/local/share/OpenCV") 
find_package(OpenCV REQUIRED)
set(OpenCV_LIB_DIR ${OpenCV_INSTALL_PATH}/lib)
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "    头文件目录：${OpenCV_INCLUDE_DIRS}")
message(STATUS "    库文件目录：${OpenCV_LIB_DIR}")
message(STATUS "    库文件列表：${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
 




CUDA_ADD_EXECUTABLE(    main_covins
                        main_covins.cpp
                        ./src/parameters.cpp
                        ./src/stereo_mapper.cpp
                        ./src/calc_cost.cu)









target_link_libraries(main_covins   ${OpenCV_LIBS} 
                            ${CUDA_LIBRARIES})

