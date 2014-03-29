//
//  DepthFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__DepthFeatureExtractor__
#define __Segmenthreetion__DepthFeatureExtractor__

#include "FeatureExtractor.h"

#include "DepthParametrization.hpp"

using namespace std;

class DepthFeatureExtractor : public FeatureExtractor
{
public:
    DepthFeatureExtractor();
	DepthFeatureExtractor(DepthParametrization dParam);
    
    void setParam(DepthParametrization dParam);

    void describe(ModalityGridData& data);
    
private:
    /*
     * Class attributes
     */
    
    DepthParametrization m_DepthParam;
    
    /*
     * Private methods
     */
    
    void describeNormalsOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tNormalsOrientsHist);
};

#endif /* defined(__Segmenthreetion__DepthFeatureExtractor__) */
