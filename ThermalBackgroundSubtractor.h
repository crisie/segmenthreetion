//
//  ThermalBackgroundSubtractor.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 07/03/14.
//
//

#ifndef __segmenthreetion__ThermalBackgroundSubtractor__
#define __segmenthreetion__ThermalBackgroundSubtractor__

#include <iostream>

#include "ModalityData.hpp"

class ThermalBackgroundSubtractor : public BackgroundSubtractor
{
    
private:
    
    //void extractItemsFromMask(cv::Mat frame, cv::Mat & output);
    
public:
    
    ThermalBackgroundSubtractor();
    
    void getMasks(ModalityData& mdInput, ModalityData& mdOutput);
    
    void getBoundingRects(ModalityData& mdInput, ModalityData& mdOutput, vector<vector<int> > &validBb);
    
    void adaptGroundTruthToReg(ModalityData& md);
    
    void getRoiTags(ModalityData& mdInput, ModalityData& mdOutput, vector<vector<int> > validBb);
};

#endif /* defined(__segmenthreetion__ThermalBackgroundSubtractor__) */
