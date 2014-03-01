//
//  FeatureExtractor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{
}

/*
 * Hypercube normalization
 */
void FeatureExtractor::hypercubeNorm(cv::Mat & src, cv::Mat & dst)
{
    src.copyTo(dst);
    double z = sum(src).val[0]; // partition function :D
    dst = dst / z;
}