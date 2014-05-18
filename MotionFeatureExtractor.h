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
    
    void describe(ModalityGridData& data);
    void describe(GridMat grid, GridMat gmask, cv::Mat gvalidness, GridMat& gdescriptors);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
    // Auxiliary
    static void computeOpticalFlow(vector<cv::Mat> colorFrames, vector<cv::Mat> & motionFrames);
    static void computeOpticalFlow(pair<cv::Mat,cv::Mat> colorFrames, cv::Mat & motionFrame);
    
private:
    MotionParametrization m_Param;
    
    void describeMotionOrientedFlow(const cv::Mat grid, const cv::Mat mask, cv::Mat & mOrientedFlowHist);
};

#endif /* defined(__segmenthreetion__MotionFeatureExtractor__) */
