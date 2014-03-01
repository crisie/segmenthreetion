//
//  ColorFeatureExtractor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#ifndef __segmenthreetion__ColorFeatureExtractor__
#define __segmenthreetion__ColorFeatureExtractor__

#include "FeatureExtractor.h"

#include "ColorParametrization.hpp"

using namespace std;

class ColorFeatureExtractor : public FeatureExtractor
{
public:
    ColorFeatureExtractor();
	ColorFeatureExtractor(ColorParametrization dParam);
    
    void setParam(ColorParametrization dParam);
    
    void describe(ModalityGridData data, GridMat& descriptors);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
private:
    /*
     * Class attributes
     */
    
    ColorParametrization m_ColorParam;
    
    /*
     * Private methods
     */
    
    void describeColorHog(const cv::Mat cell, const cv::Mat mask, cv::Mat & cOrientedGradsHist);
};

#endif /* defined(__segmenthreetion__ColorFeatureExtractor__) */
