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
    
    void getMasks(ModalityData& mdInput, ModalityData& mdOutput);
    
    void getBoundingRects(ModalityData& mdInput, ModalityData& mdOutput);
    
    void adaptGroundTruthToReg(ModalityData& mdInput, ModalityData& mdOutput);
    
    void getRoiTags(ModalityData& mdInput, ModalityData& mdOutput);
    
    void getGroundTruthBoundingRects(ModalityData& mdInput, ModalityData& mdOutput);
};

#endif /* defined(__segmenthreetion__ColorBackgroundSubtractor__) */
