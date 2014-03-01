//
//  ModalityData.h
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#ifndef __segmenthreetion__ModalityData__
#define __segmenthreetion__ModalityData__

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class ModalityData
{
public:
    ModalityData() {}
    
    ModalityData(vector<cv::Mat> frames, vector<cv::Mat> masks, vector< vector<cv::Rect> > rects, vector< vector<int> > tags) : m_Frames(frames), m_Masks(masks), m_BoundingRects(rects), m_Tags(tags) {}
    
    // Getters
    
    vector<cv::Mat>& getFrames()
    {
        return m_Frames;
    }
    
    vector<cv::Mat>& getMasks()
    {
        return m_Masks;
    }
    
    vector< vector<cv::Rect> >& getBoundingRects()
    {
        return m_BoundingRects;
    }
    
    vector< vector<int> >& getTags()
    {
        return m_Tags;
    }
    
    unsigned char getMasksOffset()
    {
        return m_MasksOffset;
    }
    
    bool isFilled()
    {
        return m_Frames.size() > 0 && m_Masks.size() > 0 && m_BoundingRects.size() > 0 && m_Tags.size() > 0;
    }
    
    // Setters
    
    void setFrames(vector<cv::Mat> frames)
    {
        m_Frames = frames;
    }
    
    void setMasks(vector<cv::Mat> masks)
    {
        m_Masks = masks;
    }
    
    void setBoundingRects(vector< vector<cv::Rect> > rects)
    {
        m_BoundingRects = rects;
    }
    
    void setTags(vector< vector<int> > tags)
    {
        m_Tags = tags;
    }
    
    void setMasksOffset(unsigned char masksOffset)
    {
        m_MasksOffset = masksOffset;
    }
    
private:
    // Load data from disk: frames, masks, and rectangular bounding boxes
    vector<cv::Mat> m_Frames;
    vector<cv::Mat> m_Masks;
    vector< vector<cv::Rect> > m_BoundingRects;
    vector< vector<int> > m_Tags;
    
    unsigned char m_MasksOffset;
};

#endif /* defined(__segmenthreetion__ModalityData__) */
