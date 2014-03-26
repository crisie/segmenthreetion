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
    
    /*ModalityGridData(vector<GridMat> gframes, vector<GridMat> gmasks, cv::Mat gframeids, cv::Mat gpositions, cv::Mat tags)
            : m_GFrames(gframes), m_GMasks(gmasks), m_GFrameIDs(gframeids), m_GPositions(gpositions), m_Tags(tags) {}
    */
    ModalityGridData(ModalityGridData& other, cv::Mat indices)
    {
        vector<GridMat> gframes;
        vector<GridMat> gmasks;
        vector<int> gframeids;
        vector<int> tags;
             
        for (int i = 0; i < other.getTags().size(); i++)
        {
            int index = (indices.rows > 1) ? indices.at<int>(i,0) : indices.at<int>(0,i);

            gframes.push_back(other.getGridFrame(index));
            gmasks.push_back(other.getGridMask(index));
			tags.push_back(other.getTag(index));
        }
        
        setGridsFrames(gframes);
        setGridsMasks(gmasks);
        setGridsFrameIDs(gframeids);
        setTags(tags);
    }

	void clear()
	{
		m_GFrames.clear();
		m_GMasks.clear();
		m_GFrameIDs.clear();
		m_FramesResolutions.clear();
		m_GBoundingRects.clear();
		m_Tags.clear();
	}
    
    void operator=(ModalityGridData& other)
    {
        m_GFrames = other.m_GFrames;
        m_GMasks = other.m_GMasks;
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
    
    int getGridFrameID(int k)
    {
        return m_GFrameIDs[k];
    }
    
    cv::Point2d getFrameResolution(int k)
    {
        return m_FramesResolutions[k];
    }
    
    cv::Rect getGridBoundingRect(int k)
    {
        return m_GBoundingRects[k];
    }
    
    int getTag(int k)
    {
        return m_Tags[k];
    }
    
    vector<GridMat>& getGridsFrames()
    {
        return m_GFrames;
    }
    
    vector<GridMat>& getGridsMasks()
    {
        return m_GMasks;
    }
    
    vector<int>& getGridsFrameIDs()
    {
        return m_GFrameIDs;
    }

    vector<cv::Point2d>& getFramesResolutions()
    {
        return m_FramesResolutions;
    }
    
    vector<cv::Rect>& getGridsBoundingRects()
    {
        return m_GBoundingRects;
    }
    
    vector<int>& getTags()
    {
        return m_Tags;
    }

	cv::Mat getGridsFrameIDsMat()
    {
		return cv::Mat(m_GFrameIDs.size(), 1, cv::DataType<int>::type, m_GFrameIDs.data());
    }
    
	cv::Mat getTagsMat()
    {
		return cv::Mat(m_Tags.size(), 1, cv::DataType<int>::type, m_Tags.data());
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
        return m_GFrames.size() > 0 && m_GMasks.size() > 0 && m_Tags.size() > 0;
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
    
    void setGridsFrameIDs(vector<int> gframeids)
    {
        m_GFrameIDs = gframeids;
    }
    
    void setFramesResolutions(vector<cv::Point2d> resolutions)
    {
        m_FramesResolutions = resolutions;
    }
    
    void setGridsBoundingRects(vector<cv::Rect> gboundingrects)
    {
        m_GBoundingRects = gboundingrects;
    }
    
    void setTags(vector<int> tags)
    {
        m_Tags = tags;
    }

    void addGridFrame(GridMat gframe)
    {
        m_GFrames.push_back(gframe);
    }
    
    void addGridMask(GridMat gmask)
    {
		m_GMasks.push_back(gmask);
    }
    
    void addGridFrameID(int id)
    {
        m_GFrameIDs.push_back(id);
    }
    
    void addFrameResolution(int x, int y)
    {
        m_FramesResolutions.push_back(cv::Point2d(x,y));
    }
    
    void addGridBoundingRect(cv::Rect gboundingrect)
    {
        m_GBoundingRects.push_back(gboundingrect);
    }
    
    void addTag(int tag)
    {
        m_Tags.push_back(tag);
    }
    
private:
    int m_hp, m_wp;
    
    vector<GridMat> m_GFrames;
    vector<GridMat> m_GMasks;
    vector<int> m_GFrameIDs;
    vector<cv::Point2d> m_FramesResolutions;
    vector<cv::Rect> m_GBoundingRects;
    vector<int> m_Tags;
};


#endif /* defined(__segmenthreetion__ModalityGridData__) */
