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
#include "ModalityGridData.hpp"

using namespace std;

class FeatureExtractor
{
public:
    FeatureExtractor();
    
    // Describe grids at cell-level
//    virtual void describe(ModalityGridData& data) = 0;
    void describe(ModalityGridData& data);
    virtual void describe(GridMat data, GridMat gmask, cv::Mat gvalidness, GridMat& descriptors) = 0;
    
protected:    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__segmenthreetion__FeatureExtractor__) */
