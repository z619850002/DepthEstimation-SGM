//main.cpp
#include <iostream>
#include <vector>
#include <fstream>


// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/eigen.hpp>

#include "./include/parameters.h"
#include "./include/stereo_mapper.h"
#include "./include/calc_cost.h"

#include <boost/timer.hpp>







using namespace std;

const int width = 752;      
const int height = 480;     
const double fx = 4.616e+02;   
const double fy = 4.603e+02;
const double cx = 3.630e+02;
const double cy = 2.481e+02;


cv::Mat mK1 = (cv::Mat_<double>(3 , 3) <<   fx ,  0.0 , cx,
                                            0.0,  fy  , cy,
                                            0.0,  0.0 , 1.0);
cv::Mat mD1 = (cv::Mat_<double>(5 , 1) << -2.917e-01, 8.228e-02, 5.333e-05, -1.578e-04, 0.0);



cv::Mat mRefRotation_wc = (cv::Mat_<double>(3 , 3) <<   -0.9167014887227345, 0.09766387450836564, -0.3874534142184533,
                                                         0.3961614620193463, 0.3485840136993414, -0.8494382151775085,
                                                         0.05210063900007567, -0.9321753874723745, -0.3582381476148404);

cv::Mat mRefTranslation_wc = (cv::Mat_<double>(3 , 1) << -3.72535, -5.87637, -0.11411);

cv::Mat mCurrRotation_wc = (cv::Mat_<double>(3 , 3) <<   -0.9142634355785113, 0.1105007951715473, -0.3897588288039585,
                                                         0.4021692615364046, 0.3634688867660582, -0.8403274679720433,
                                                         0.04880835419721374, -0.9250296982364935, -0.37674633633396);

cv::Mat mCurrTranslation_wc = (cv::Mat_<double>(3, 1) << -3.85745, -5.83009, -0.07885209999999999);


int main (int argc, char* argv[])
{
    cv::Mat mRefImage = cv::imread("./sample/ref.jpg" , 0);    
    cv::Mat mCurrImage = cv::imread("./sample/match.jpg" , 0);
        

#if DOWNSAMPLE
    cv::resize(mRefImage, mRefImage, cv::Size(width/2, height/2));
    cv::resize(mCurrImage, mCurrImage, cv::Size(width/2, height/2));
#endif
    
    StereoMapper iMapper = StereoMapper();
    iMapper.initIntrinsic(mK1, mD1, mK1, mD1);

    
    iMapper.initReference(mRefImage);

    iMapper.update( mCurrImage,
                    mRefRotation_wc, 
                    mRefTranslation_wc, 
                    mCurrRotation_wc, 
                    mCurrTranslation_wc);


    cv::Mat mResultMap = iMapper.output();

#if DOWNSAMPLE
        cv::resize(mResultMap, mResultMap, cv::Size(width, height));

#endif


    cv::imshow("depth", mResultMap*0.1);
    cv::waitKey(0);

    return 0;
}

