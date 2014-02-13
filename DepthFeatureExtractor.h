//
//  DepthFeatureExtractor.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__DepthFeatureExtractor__
#define __Segmenthreetion__DepthFeatureExtractor__

#include <opencv2/opencv.hpp>

#include <iostream>

#include "GridMat.h"
#include "DepthParametrization.h"

using namespace std;

class DepthFeatureExtractor
{
public:
    DepthFeatureExtractor(int hp, int wp);
	DepthFeatureExtractor(int hp, int wp, DepthParametrization dParam);
    
    void setData(vector<GridMat> grids, vector<GridMat> masks);
    void setParam(DepthParametrization dParam);

    void describe(GridMat & descriptions);
    void describe(GridMat & subjectDescriptions, GridMat & objectDescriptions);
    
private:
    /*
     * Class attributes
     */
    
    int m_hp;
    int m_wp;
    
    vector<GridMat> m_DepthGrids;
    
    vector<GridMat> m_DepthMasks;
    
    DepthParametrization m_DepthParam;

    GridMat m_DepthDescriptions;
    
    /*
     * Private methods
     */
    
    void describeNormalsOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tNormalsOrientsHist);
    
    // Normalize a descriptor (hypercube, i.e. f: (-inf, inf) --> [0, 1]
    void hypercubeNorm(cv::Mat & src, cv::Mat & dst);
};

#endif /* defined(__Segmenthreetion__DepthFeatureExtractor__) */
