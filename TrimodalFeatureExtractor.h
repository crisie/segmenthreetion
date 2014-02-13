//
//  TrimodalFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__TrimodalFeatureExtractor__
#define __Segmenthreetion__TrimodalFeatureExtractor__

#include <opencv2/opencv.hpp>

#include <iostream>

#include "GridMat.h"
#include "ThermalParametrization.h"
#include "DepthParametrization.h"
#include "ColorParametrization.h"

using namespace std;

class TrimodalFeatureExtractor
{
public:
    TrimodalFeatureExtractor(int hp, int wp);
    
    void setThermalData(vector<GridMat> grids, vector<GridMat> masks);
    void setThermalParam(ThermalParametrization tParam);
    
    void setDepthData(vector<GridMat> grids, vector<GridMat> masks);
    void setDepthParam(DepthParametrization dParam);

    void describe(GridMat & descriptionsThermal, GridMat & descriptionsDepth);
    
private:
    /*
     * Class attributes
     */
    
    int m_hp;
    int m_wp;
    
    vector<GridMat> m_ThermalGrids;
    vector<GridMat> m_DepthGrids;
//    vector<GridMat> m_ColorGrids;
    
    vector<GridMat> m_ThermalMasks;
    vector<GridMat> m_DepthMasks;
//    vector<GridMat> m_ColorFrames;
    
    ThermalParametrization m_ThermalParam;
    DepthParametrization m_DepthParam;
//    ColorParametrization m_ColorParam;
    
    GridMat m_ThermalDescriptions;
    GridMat m_DepthDescriptions;
//    GridMat m_ColorDescriptions;
    
    /*
     * Private methods
     */
    
    void describeThermal(GridMat & descriptions);
    void describeDepth(GridMat & descriptions);
    
    void describeThermalIntesities(const cv::Mat grid, const cv::Mat mask, cv::Mat & tIntensityHist);
    void describeThermalGradOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tGradOrientsHist);
    void describeNormalsOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tNormalsOrientsHist);
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__Segmenthreetion__TrimodalFeatureExtractor__) */
