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

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GridMat.h"
#include "ColorParametrization.hpp"

using namespace std;

class ColorFeatureExtractor : public FeatureExtractor
{
public:
    ColorFeatureExtractor(int hp, int wp);
	ColorFeatureExtractor(int hp, int wp, ColorParametrization dParam);
    
    void setParam(const ColorParametrization dParam);
    
    void describe(vector<GridMat> grids, vector<GridMat> masks, GridMat & descriptors);
    void describe(vector<GridMat> grids, vector<GridMat> masks,
                  GridMat & subDescriptors, GridMat & objDescriptors, GridMat & unkDescriptors);
    
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
