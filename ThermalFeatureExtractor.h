//
//  ThermalFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__ThermalFeatureExtractor__
#define __Segmenthreetion__ThermalFeatureExtractor__

#include "FeatureExtractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "GridMat.h"
#include "ThermalParametrization.hpp"

using namespace std;

class ThermalFeatureExtractor : public FeatureExtractor
{
public:
    ThermalFeatureExtractor(int hp, int wp);
	ThermalFeatureExtractor(int hp, int wp, ThermalParametrization tParam);
    
    void setParam(ThermalParametrization tParam);

    void describe(vector<GridMat> grids, vector<GridMat> masks, GridMat & descriptors);
    void describe(vector<GridMat> grids, vector<GridMat> masks,
                  GridMat & subDescriptors, GridMat & objDescriptors, GridMat & unkDescriptors);
    
private:
    /*
     * Class attributes
     */
    
    ThermalParametrization m_ThermalParam;
    
    /*
     * Private methods
     */
    
    void describeThermalIntesities(const cv::Mat grid, const cv::Mat mask, cv::Mat & tIntensityHist);
    void describeThermalGradOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tGradOrientsHist);
};

#endif /* defined(__Segmenthreetion__ThermalFeatureExtractor__) */
