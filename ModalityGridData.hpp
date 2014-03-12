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
    
    ModalityGridData(vector<GridMat> gframes, vector<GridMat> gmasks, vector<GridMat> gpredmasks, cv::Mat gframeids, cv::Mat gpositions, cv::Mat tags)
            : m_GFrames(gframes), m_GMasks(gmasks), m_GPredictedMasks(gpredmasks), m_GFrameIDs(gframeids), m_GPositions(gpositions), m_Tags(tags) {}
    
    ModalityGridData(ModalityGridData& other, cv::Mat indices)
    {
        vector<GridMat> gframes;
        vector<GridMat> gmasks;
        vector<GridMat> gpredmasks;
        cv::Mat gframeids;
        cv::Mat gpositions;
        cv::Mat tags (cv::sum(indices).val[0], other.getTags().cols, other.getTags().type());
        
        int ntags = (other.getTags().rows > 1) ? other.getTags().rows : other.getTags().cols;        
        for (int i = 0; i < ntags; i++)
        {
            int index = (indices.rows > 1) ? indices.at<int>(i,0) : indices.at<int>(0,i);

            gframes.push_back(other.getGridFrame(index));
            gmasks.push_back(other.getGridMask(index));
            gpredmasks.push_back(other.getGridPredictedMask(index));
            
            if (other.getTags().rows > 1)
                tags.at<int>(i,0) = other.getTag(index);
            else
                tags.at<int>(0,i) = other.getTag(index);
            
        }
        
        setGridsFrames(gframes);
        setGridsMasks(gmasks);
        setGridsPredictedMasks(gpredmasks);
        setGridsFrameIDs(gframeids);
        //segGridsPositions(gpositions);
        setTags(tags);
    }
    
    void operator=(ModalityGridData& other)
    {
        m_GFrames = other.m_GFrames;
        m_GMasks = other.m_GMasks;
        m_GPredictedMasks = other.m_GPredictedMasks;
        m_Tags = other.m_Tags;
    }
    
    // Getters
    
    GridMat getGridFrame(int k)
    {
        return m_GFrames[k];
    }
    
    GridMat getGridMask(int k)
    {
        return m_GMasks[k];
    }
    
    GridMat getGridPredictedMask(int k)
    {
        return m_GPredictedMasks[k];
    }
    
    int getGridFrameID(int k)
    {
        return m_GFrameIDs.at<int>(k,0);
    }
    
    cv::Point2d getFrameResolution(int k)
    {
        return cv::Point2d(m_FramesResolutions.at<int>(k,0), m_FramesResolutions.at<int>(k,1));
    }
    
    cv::Rect getGridBoundingRect(int k)
    {
        return m_GBoundingRects[k];
    }
    
    int getTag(int k)
    {
        return (m_Tags.rows > 1) ? m_Tags.at<int>(k,0) : m_Tags.at<int>(0,k);
    }
    
    vector<GridMat>& getGridsFrames()
    {
        return m_GFrames;
    }
    
    vector<GridMat>& getGridsMasks()
    {
        return m_GMasks;
    }
    
    vector<GridMat>& getGridsPredictedMasks()
    {
        return m_GPredictedMasks;
    }
    
    cv::Mat& getGridsFrameIDs()
    {
        return m_GFrameIDs;
    }

    cv::Mat& getFramesResolutions()
    {
        return m_FramesResolutions;
    }
    
    vector<cv::Rect>& getGridsBoundingRects()
    {
        return m_GBoundingRects;
    }
    
    cv::Mat& getTags()
    {
        return m_Tags;
    }
    
    int hp()
    {
        return m_hp;
    }
    
    int wp()
    {
        return m_wp;
    }
    
    bool isFilled()
    {
        return m_GFrames.size() > 0 && m_GMasks.size() > 0 && m_GPredictedMasks.size() > 0 && (m_Tags.rows > 1 || m_Tags.cols > 1);
    }
    
    // Setters
    
    void setGridsFrames(vector<GridMat> gframes)
    {
        m_GFrames = gframes;
    }
    
    void setGridsMasks(vector<GridMat> gmasks)
    {
        m_GMasks = gmasks;
    }
    
    void setGridsPredictedMasks(vector<GridMat> gmasks)
    {
        m_GPredictedMasks = gmasks;
    }
    
    void setGridsFrameIDs(cv::Mat gframeids)
    {
        m_GFrameIDs = gframeids;
    }
    
    void setFramesResolutions(cv::Mat resolutions)
    {
        m_FramesResolutions = resolutions;
    }
    
    void setGridsBoundingRects(vector<cv::Rect> gboundingrects)
    {
        m_GBoundingRects = gboundingrects;
    }
    
    void setTags(cv::Mat tags)
    {
        m_Tags = tags;
    }
    
private:
    int m_hp, m_wp;
    
    vector<GridMat> m_GFrames;
    vector<GridMat> m_GMasks;
    vector<GridMat> m_GPredictedMasks;
    cv::Mat m_GFrameIDs;
    cv::Mat m_FramesResolutions;
    vector<cv::Rect> m_GBoundingRects;
    cv::Mat m_Tags;
};


#endif /* defined(__segmenthreetion__ModalityGridData__) */
