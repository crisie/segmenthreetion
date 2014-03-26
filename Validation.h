//
//  Validation.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 21/03/14.
//
//

#ifndef __segmenthreetion__Validation__
#define __segmenthreetion__Validation__

#include <iostream>
#include <vector>
#include <set>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;


class Validation {

public:
    Validation();
    
    void getOverlap(vector<cv::Mat> predictedMasks, vector<cv::Mat> gtMasks, vector<int> dcRange, cv::Mat& overlapIDs);
    
private:
  
    float getMaskOverlap(cv::Mat predictedMask, cv::Mat gtMask, cv::Mat dontCareRegion);
    
    void createDontCareRegion(cv::Mat inputMask, cv::Mat& outputMask, float dcRange);
};



#endif /* defined(__segmenthreetion__Validation__) */
