//
//  TrimodalClusterer.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__TrimodalClusterer__
#define __Segmenthreetion__TrimodalClusterer__

#include "GridMat.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class TrimodalClusterer
{
public:
    /*
     * Constructors
     */
    TrimodalClusterer(int C);
    
    /*
     * Public methods
     */
    
    // Clusterize thermal images (roughly described), and get a label per image
    // poseLabels: vector of as many elements as grid cells, and the mat is also 1-D matrix of §length number of grids
//    void clusterize(vector<GridMat> grids, vector<GridMat> gridsMasks, Mat indices, vector<Mat> & cellLabels, vector<Mat> & cellCenters);
//    Mat clusterize(GridMat & grid, GridMat & mask, vector<Mat> cellCenters);
    
    void trainClusters(GridMat descriptors, GridMat & labels, GridMat & centers); // in training
    void predictClusters(GridMat descriptors, GridMat centers, GridMat & labels);   // in testint (prediction)
    
private:
    /*
     * Class attributes
     */
    unsigned int    m_hp; // Num of grid paritions along the height (rows or y-direction)
    unsigned int    m_wp; // Num of grid partitions along the width (cols or x-direction)
    double          m_kparam; // Relative size of the convolving kernel
    unsigned int    m_bins; // Number of bins of the histogram describing a cell in the grid
    int             m_C; // Num clusters
    
    /*
     * Private methods
     */

    // Closest cluster center at a point
    int closestClusterCenter(cv::Mat point, cv::Mat centers);
};

#endif /* defined(__Segmenthreetion__TrimodalClusterer__) */
