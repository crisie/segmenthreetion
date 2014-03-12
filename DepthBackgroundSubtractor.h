//
//  DepthBackgroundSubtractor.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 05/03/14.
//
//

#ifndef __segmenthreetion__DepthBackgroundSubtractor__
#define __segmenthreetion__DepthBackgroundSubtractor__

#include <iostream>

#include "BackgroundSubtractor.h"
#include "ModalityData.hpp"
#include "ForegroundParametrization.hpp"

class DepthBackgroundSubtractor : public BackgroundSubtractor {

private:
    
    //unsigned char m_masksOffset;
    ForegroundParametrization m_fParam;
    
    void extractItemsFromMask(cv::Mat frame, cv::Mat & output);
    
public:
    
    DepthBackgroundSubtractor();
    
    DepthBackgroundSubtractor(ForegroundParametrization fParam);
    
    //DepthBackgroundSubtractor(vector<int> numFramesToLearn, unsigned char masksOffset);
    
    //void setNumFramesToLearn(vector<int> numFramesToLearn);
    
    //void setMasksOffset(unsigned char masksOffset);
    
    void getMasks(ModalityData& md);
    
    void getBoundingRects(ModalityData& md);
    
    void adaptGroundTruthToReg(ModalityData& md);
    
};

#endif /* defined(__segmenthreetion__DepthBackgroundSubtractor__) */
