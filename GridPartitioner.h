//
//  GridPartitioner.h
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#ifndef __segmenthreetion__GridPartitioner__
#define __segmenthreetion__GridPartitioner__

#include <iostream>

#include "GridMat.h"
#include "ModalityData.hpp"
#include "ModalityGridData.hpp"

class GridPartitioner
{
public:
    
    GridPartitioner(); // default: 2 x 2
    GridPartitioner(unsigned int hp, unsigned int wp);
    
    void setGridPartitions(unsigned int hp, unsigned int wp);

    void grid(ModalityData& md, ModalityGridData& mgd);
    
private:
    
    unsigned int m_hp, m_wp; // partitions in height and width
    
    // Trim subimages (using the rects provided) from frames
    void gridFrames(ModalityData& md, vector<GridMat>& gframes, cv::Mat& frameIDs);
    void gridMasks(ModalityData& md, vector<GridMat>& gmasks);
    void gridMasks(ModalityData& md, vector<GridMat>& gmasks, vector<cv::Rect>& grects, cv::Mat& gtags);
};

#endif /* defined(__segmenthreetion__GridPartitioner__) */
