//
//  MotionFeatureExtractor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#ifndef __segmenthreetion__MotionFeatureExtractor__
#define __segmenthreetion__MotionFeatureExtractor__

#include "FeatureExtractor.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GridMat.h"
#include "MotionParametrization.hpp"

using namespace std;

class MotionFeatureExtractor : public FeatureExtractor
{
public:
    MotionFeatureExtractor(const unsigned int hp, const unsigned int wp);
	MotionFeatureExtractor(const unsigned int hp, const unsigned int wp, MotionParametrization mParam);
    
    void setParam(const MotionParametrization param);
    
    void describe(vector<GridMat> grids, vector<GridMat> masks, GridMat & descriptors);
    void describe(vector<GridMat> grids, vector<GridMat> masks,
                  GridMat & subDescriptors, GridMat & objDescriptors, GridMat & unkDescriptors);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
private:
    MotionParametrization m_Param;
    
    void describeMotionOrientedFlow(const cv::Mat grid, const cv::Mat mask, cv::Mat & mOrientedFlowHist);
    
    // Auxiliary
    void computeOpticalFlow(vector<cv::Mat> colorFrames, vector<cv::Mat> & motionFrames);
};

#endif /* defined(__segmenthreetion__MotionFeatureExtractor__) */
