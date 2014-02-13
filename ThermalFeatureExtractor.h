//
//  ThermalFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__ThermalFeatureExtractor__
#define __Segmenthreetion__ThermalFeatureExtractor__

#include <opencv2/opencv.hpp>

#include <iostream>

#include "GridMat.h"
#include "ThermalParametrization.h"

using namespace std;

class ThermalFeatureExtractor
{
public:
    ThermalFeatureExtractor(int hp, int wp);
	ThermalFeatureExtractor(int hp, int wp, ThermalParametrization tParam);
    
    void setData(vector<GridMat> grids, vector<GridMat> masks);
    void setParam(ThermalParametrization tParam);

    void describe(GridMat & descriptionsThermal);
    void describe(GridMat & subjectDescriptors, GridMat & objectDescriptors);
    
private:
    /*
     * Class attributes
     */
    
    int m_hp;
    int m_wp;
    
    vector<GridMat> m_ThermalGrids;
    vector<GridMat> m_ThermalMasks;
    ThermalParametrization m_ThermalParam;
    
    GridMat m_ThermalDescriptions;
    
    /*
     * Private methods
     */
    
    void describeThermalIntesities(const cv::Mat grid, const cv::Mat mask, cv::Mat & tIntensityHist);
    void describeThermalGradOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tGradOrientsHist);
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__Segmenthreetion__ThermalFeatureExtractor__) */
