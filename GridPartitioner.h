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
    GridPartitioner(unsigned int hp, unsigned int wp, ModalityData md);
    
    void setModalityData(ModalityData md);
    
    void setGridPartitions(unsigned int hp, unsigned int wp);
    
    void grid(ModalityGridData& mgd);
    void grid(ModalityData md, ModalityGridData& mgd);
    
private:
    
    unsigned int m_hp, m_wp; // partitions in height and width
    ModalityData m_md;
    ModalityGridData m_mgd;
    
    // Trim subimages (using the rects provided) from frames
    void grid(vector<cv::Mat>& images, vector< vector<cv::Rect> > rects, unsigned int crows, unsigned int ccols, vector<GridMat>& grids);
    void grid(vector<cv::Mat>& images, vector< vector<cv::Rect> > rects, vector< vector<int> > rtags, unsigned int crows, unsigned int ccols, vector<GridMat>& grids, cv::Mat& tags);
};

#endif /* defined(__segmenthreetion__GridPartitioner__) */
