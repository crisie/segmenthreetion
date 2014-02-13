//
//  ColorFeatureExtractor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#ifndef __segmenthreetion__ColorFeatureExtractor__
#define __segmenthreetion__ColorFeatureExtractor__

#include <iostream>

#include <opencv2/opencv.hpp>

#include "GridMat.h"
#include "ColorParametrization.h"

using namespace std;

class ColorFeatureExtractor
{
public:
    ColorFeatureExtractor(int hp, int wp);
	ColorFeatureExtractor(int hp, int wp, ColorParametrization dParam);
    
    void setData(vector<GridMat> grids, vector<GridMat> masks);
    void setParam(ColorParametrization dParam);
    
    void describe(GridMat & descriptions);
    void describe(GridMat & subjectDescriptions, GridMat & objectDescriptions);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
private:
    /*
     * Class attributes
     */
    
    int m_hp;
    int m_wp;
    
    vector<GridMat> m_ColorGrids;
    
    vector<GridMat> m_ColorMasks;
    
    ColorParametrization m_ColorParam;
    
    GridMat m_ColorDescriptions;
    
    /*
     * Private methods
     */
    
    void describeColor(GridMat & descriptors);
    void describeColorHog(const cv::Mat grid, const cv::Mat mask, cv::Mat & cOrientedGradsHist);
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__segmenthreetion__ColorFeatureExtractor__) */
