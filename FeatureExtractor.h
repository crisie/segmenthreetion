//
//  FeatureExtractor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#ifndef __segmenthreetion__FeatureExtractor__
#define __segmenthreetion__FeatureExtractor__

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GridMat.h"

using namespace std;

class FeatureExtractor
{
public:
    FeatureExtractor(const unsigned int hp, const unsigned int wp);
    
    // Describe grids at cell-level
    virtual void describe(vector<GridMat>& grids, vector<GridMat>& gmasks, GridMat & descriptions) = 0;
    
protected:
    const unsigned int m_hp;
    const unsigned int m_wp;
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__segmenthreetion__FeatureExtractor__) */
