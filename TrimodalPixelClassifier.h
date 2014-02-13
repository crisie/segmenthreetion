//
//  TrimodalPixelClassifier.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 21/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__TrimodalPixelClassifier__
#define __Segmenthreetion__TrimodalPixelClassifier__

#include "GridMat.h"

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "em30.h"

class TrimodalPixelClassifier
{

public:
    /*
     * Constructors
     */
    TrimodalPixelClassifier(unsigned int classes);
    
    /*
     * Public methods
     */
    void train(GridMat descriptions, GridMat labels, GridMat centers);
    void test(GridMat descriptors, GridMat labels, GridMat & classes, 
		GridMat & probabilities, GridMat & logLikelihoods);
    void test2(GridMat descriptors, GridMat labels, GridMat & classes,
              GridMat & probabilities, GridMat & logLikelihoods);
    
private:
    /*
     * Class attributes
     */
    double m_classes;
    
    GridMat m_TrainDescriptions;
    GridMat m_TrainLabels;
    GridMat m_TrainCenters;
    
    std::vector< std::vector<cv::EM> > m_TEMMatrix;
    
    /*
     * Private methods
     */
    void describeThermalCell(const cv::Mat & cell, const cv::Mat & mask, cv::Mat & tDescriptorsMatrix);
	void modelLogLikelihoodsPDFs(const GridMat descriptors, const GridMat labels);
};

#endif /* defined(__Segmenthreetion__TrimodalPixelClassifier__) */
