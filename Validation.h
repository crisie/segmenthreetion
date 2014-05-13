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

#include "ModalityData.hpp"
#include "ModalityGridData.hpp"

using namespace std;


class Validation {

public:
    Validation();
    
    Validation(vector<int> dcRange);
    
    void setDontCareRange(vector<int> dcRange);
    
    vector<int>& getDontCareRange();
    
    void getOverlap(ModalityData& md, cv::Mat& overlapIDs);
    
    void getOverlap(vector<cv::Mat>& predictedMasks, vector<cv::Mat>& gtMasks, vector<int>& dcRange, cv::Mat& overlapIDs);
    
    void getMeanOverlap(vector<cv::Mat> partitionedOverlapIDs, cv::Mat& partitionedMeanOverlap);
    
    void createOverlapPartitions(cv::Mat& partitions, cv::Mat& overlapIDs, vector<cv::Mat>& partitionedOverlapIDs);
    
    void save(vector<cv::Mat> overlapIDs, cv::Mat meanOverlap, string filename);
    
private:
  
    vector<int> m_DontCareRange;
    
    float getMaskOverlap(cv::Mat& predictedMask, cv::Mat& gtMask, cv::Mat& dontCareRegion);
    
    void createDontCareRegion(cv::Mat& inputMask, cv::Mat& outputMask, int dcRange);
};



#endif /* defined(__segmenthreetion__Validation__) */
