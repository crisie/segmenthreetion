//
//  MotionFeatureExtractor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#ifndef __segmenthreetion__MotionFeatureExtractor__
#define __segmenthreetion__MotionFeatureExtractor__

#include <iostream>

#include <opencv2/opencv.hpp>

#include "GridMat.h"
#include "MotionParametrization.h"

using namespace std;

class MotionFeatureExtractor
{
public:
    MotionFeatureExtractor(int hp, int wp);
	MotionFeatureExtractor(int hp, int wp, MotionParametrization dParam);
    
    void setData(vector<GridMat> grids, vector<GridMat> masks);
    void setParam(MotionParametrization dParam);
    
    void describe(GridMat & descriptions);
    void describe(GridMat & subjectDescriptions, GridMat & objectDescriptions);
    
    cv::Mat get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues);
    
private:
    /*
     * Class attributes
     */
    
    int m_hp;
    int m_wp;
    
    vector<GridMat> m_MotionGrids;
    vector<GridMat> m_MotionMasks;
    
    MotionParametrization m_MotionParam;
    
    GridMat m_MotionDescriptions;
    
    /*
     * Private methods
     */
    
    void describeMotion(GridMat & descriptions);
    void describeMotionOrientedFlow(const cv::Mat grid, const cv::Mat mask, cv::Mat & mOrientedFlowHist);
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__segmenthreetion__MotionFeatureExtractor__) */
