//
//  DepthFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clap�s on 24/05/13.
//  Copyright (c) 2013 Albert Clap�s. All rights reserved.
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
    void describe(GridMat grid, GridMat gmask, cv::Mat gvalidness, GridMat& gdescriptors);
    
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
