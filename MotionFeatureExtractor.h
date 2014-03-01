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

#include "MotionParametrization.hpp"

using namespace std;

class MotionFeatureExtractor : public FeatureExtractor
{
public:
    MotionFeatureExtractor();
	MotionFeatureExtractor(MotionParametrization mParam);
    
    void setParam(MotionParametrization param);
    
    void describe(ModalityGridData data, GridMat& descriptors);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
private:
    MotionParametrization m_Param;
    
    void describeMotionOrientedFlow(const cv::Mat grid, const cv::Mat mask, cv::Mat & mOrientedFlowHist);
    
    // Auxiliary
    void computeOpticalFlow(vector<cv::Mat> colorFrames, vector<cv::Mat> & motionFrames);
};

#endif /* defined(__segmenthreetion__MotionFeatureExtractor__) */
