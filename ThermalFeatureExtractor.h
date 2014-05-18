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

#include "ThermalParametrization.hpp"

using namespace std;

class ThermalFeatureExtractor : public FeatureExtractor
{
public:
    ThermalFeatureExtractor();
	ThermalFeatureExtractor(ThermalParametrization tParam);
    
    void setParam(ThermalParametrization tParam);

    void describe(ModalityGridData& data);
    void describe(GridMat grid, GridMat gmask, cv::Mat gvalidness, GridMat& gdescriptors);
    
private:
    /*
     * Class attributes
     */
    
    ThermalParametrization m_ThermalParam;
    
    /*
     * Private methods
     */
    
    void describeThermalIntesities(cv::Mat grid, cv::Mat mask, cv::Mat & tIntensityHist);
    void describeThermalGradOrients(cv::Mat grid, cv::Mat mask, cv::Mat & tGradOrientsHist);
};

#endif /* defined(__Segmenthreetion__ThermalFeatureExtractor__) */
