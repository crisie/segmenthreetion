//
//  ModalityGridData.h
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#ifndef __segmenthreetion__ModalityGridData__
#define __segmenthreetion__ModalityGridData__

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GridMat.h"

using namespace std;

class ModalityGridData
{
public:
    ModalityGridData() {}
    
    ModalityGridData(vector<GridMat> gframes, vector<GridMat> gmasks, cv::Mat tags)
            : m_GFrames(gframes), m_GMasks(gmasks), m_Tags(tags) {}

    // Getters
    
    vector<GridMat>& getGridFrames()
    {
        return m_GFrames;
    }
    
    vector<GridMat>& getGridMasks()
    {
        return m_GMasks;
    }
    
    cv::Mat& getTags()
    {
        return m_Tags;
    }
    
    unsigned int hp()
    {
        return m_hp;
    }
    
    unsigned int wp()
    {
        return m_wp;
    }
    
    bool isFilled()
    {
        return m_GFrames.size() > 0 && m_GMasks.size() > 0 && (m_Tags.rows > 1 || m_Tags.cols > 1);
    }
    
    // Setters
    
    void setGridFrames(vector<GridMat> gframes)
    {
        m_GFrames = gframes;
    }
    
    void setGridMasks(vector<GridMat> gmasks)
    {
        m_GMasks = gmasks;
    }
    
    void setTags(cv::Mat tags)
    {
        m_Tags = tags;
    }
    
private:
    unsigned int m_hp, m_wp;
    
    vector<GridMat> m_GFrames;
    vector<GridMat> m_GMasks;
    cv::Mat m_Tags;
};


#endif /* defined(__segmenthreetion__ModalityGridData__) */
