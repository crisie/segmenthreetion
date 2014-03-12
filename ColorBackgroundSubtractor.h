//
//  ColorBackgroundSubtractor.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 07/03/14.
//
//

#ifndef __segmenthreetion__ColorBackgroundSubtractor__
#define __segmenthreetion__ColorBackgroundSubtractor__

#include <iostream>

#include "ModalityData.hpp"

class ColorBackgroundSubtractor : public BackgroundSubtractor
{
   
private:
    
    //unsigned char m_masksOffset;
    
    void extractItemsFromMask(cv::Mat frame, cv::Mat & output);
    
public:
    
    ColorBackgroundSubtractor();
    
    //void setMasksOffset(unsigned char masksOffset);
    
    void getMasks(ModalityData& mdOutput, ModalityData& mdInput);
    
    void getBoundingRects(ModalityData& mdOutput, ModalityData& mdInput);
    
    void adaptGroundTruthToReg(ModalityData& mdOutput, ModalityData& mdInput);
    
    void getRoiTags(ModalityData& mdOutput, ModalityData& mdInput);
    
    void getGroundTruthBoundingRects(ModalityData& mdOutput, ModalityData& mdInput);
};

#endif /* defined(__segmenthreetion__ColorBackgroundSubtractor__) */
