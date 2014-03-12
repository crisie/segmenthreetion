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
    
    ModalityData(vector<cv::Mat> frames, vector<cv::Mat> masks, vector<cv::Mat> predictedMasks vector< vector<cv::Rect> > rects, vector< vector<int> > tags) : m_Frames(frames), m_GroundTruthMasks(masks), m_PredictedMasks(predictedMasks), m_PredictedBoundingRects(rects), m_Tags(tags) {}
    
    // Getters
    
    cv::Mat getFrame(int k)
    {
        return m_Frames[k];
    }
    
    
    cv::Mat getPredictedMask(int k)
    {
        return m_PredictedMasks[k];
    }
    
    cv::Mat getPredictedMask(int k, unsigned char subjectId)
    {
        return (m_PredictedMasks[k] == (m_MasksOffset + subjectId));
    }
    
    /*
    int getFrameID(int k)
    {
        return m_FrameIDs[k];
    }
     */
        
    cv::Mat getGroundTruthMask(int k)
    {
        return m_GroundTruthMasks[k];
    }
    
    vector<cv::Mat> getPredictedMasksInScene(int s)
    {
        vector<cv::Mat> sceneMasks;
        
        for(int k = m_SceneLimits[s].first; k < (m_SceneLimits[s].second); k++ )
        {
            sceneMasks.push_back(m_PredictedMasks[k]);
        }
        
        return sceneMasks;
    }
    
    cv::Mat getPredictedMaskInScene(int s, int k)
    {
        return m_PredictedMasks[m_SceneLimits[s].first + k];
    }
    
    cv::Mat getPredictedMaskInScene(int s, int k, unsigned char subjectId)
    {
        return (m_PredictedMasks[m_SceneLimits[s].first + k] == (m_MasksOffset + subjectId));
    }
    
    
    vector<cv::Mat> getFramesInScene(int s)
    {
        vector<cv::Mat> sceneFrames;
        
        for(int k = m_SceneLimits[s].first; k <= (m_SceneLimits[s].second); k++ )
        {
            sceneFrames.push_back(m_Frames[k]);
        }
        
        return sceneFrames;
    }
    
    cv::Mat getFrameInScene(int s, int k)
    {
        return m_Frames[m_SceneLimits[s].first + k];
    }
    
    vector<cv::Mat> getGroundTruthMasksInScene(int s)
    {
        vector<cv::Mat> gtMasks;
        
        for(int k = m_SceneLimits[s].first; k <= (m_SceneLimits[s].second); k++ )
        {
            gtMasks.push_back(m_GroundTruthMasks[k]);
        }
        
        return gtMasks;
    }
    
    cv::Mat getGroundTruthMaskInScene(int s, int k)
    {
        return m_GroundTruthMasks[m_SceneLimits[s].first + k];
    }
    
    vector<cv::Rect> getPredictedBoundingRectsInFrame(int k)
    {
        return m_PredictedBoundingRects[k];
    }
    
    vector<cv::Rect> getGroundTruthBoundingRectsInFrame(int k)
    {
        return m_GroundTruthBoundingRects[k];
    }
    
    int getNumScenes()
    {
        return m_SceneLimits.size();
    }
    
    vector<int> getTagsInFrame(int k)
    {
        return m_Tags[k];
    }
    
    vector<cv::Mat>& getFrames()
    {
        return m_Frames;
    }
    
    vector<cv::Mat>& getPredictedMasks()
    {
        return m_PredictedMasks;
    }
    
   /*
    vector<int>& getFrameIDs()
    {
        return m_FrameIDs;
    }
    */
    vector<cv::Mat>& getGroundTruthMasks()
    {
        return m_GroundTruthMasks;
    }
    
    vector< vector<cv::Rect> >& getPredictedBoundingRects()
    {
        return m_PredictedBoundingRects;
    }
    
    vector< vector<cv::Rect> >& getGroundTruthBoundingRects()
    {
        return m_GroundTruthBoundingRects;
    }
    
    vector< vector<int> >& getTags()
    {
        return m_Tags;
    }
    
    unsigned char getMasksOffset()
    {
        return m_MasksOffset;
    }
    
    vector<string>& getCalibVarsDirs()
    {
        return m_CalibVarsDirs;
    }
    
    vector<string>& getFramesIndices()
    {
        return m_FramesIndices;
    }
    
    vector<string> getFramesIndicesInScene(int s)
    {
        vector<string> indices;
        
        for(int k = m_SceneLimits[s].first; k <= (m_SceneLimits[s].second); k++ )
        {
            indices.push_back(m_FramesIndices[k]);
        }
        
        return indices;
    }
    
    bool isFilled()
    {
        return m_Frames.size() > 0 && m_GroundTruthMasks.size() > 0 && m_PredictedMasks.size() > 0
                && m_FramesIndices.size() > 0 && m_PredictedBoundingRects.size() > 0 && m_Tags.size() > 0;
    }
    
    // Setters
    
    void setFrames(vector<cv::Mat> frames)
    {
        m_Frames = frames;
        m_FramesResolutions.create(m_Frames.size(), 2, cv::DataType<int>::type);
        for (int i = 0; i < m_Frames.size(); i++)
        {
            m_FramesResolutions.at<int>(i,0) = m_Frames[i].cols; // inverted
            m_FramesResolutions.at<int>(i,1) = m_Frames[i].rows;
        }
    }
    
    void setGroundTruthMasks(vector<cv::Mat> masks)
    {
        m_GroundTruthMasks = masks;
    }
    
    void setPredictedMasks(vector<cv::Mat> masks)
    {
        m_PredictedMasks = masks;
    }
    
    /*
    void setFrameIDs(vector<int> frameids)
    {
        m_FrameIDs = frameids;
    }
     */
    
    void setPredictedBoundingRects(vector< vector<cv::Rect> > rects)
    {
        m_PredictedBoundingRects = rects;
    }
    
    void setGroundTruthBoundingRects(vector< vector<cv::Rect> > groundTruthBoundingRects)
    {
        m_GroundTruthBoundingRects = groundTruthBoundingRects;
    }
    
    void setTags(vector< vector<int> > tags)
    {
        m_Tags = tags;
    }
    
    void setMasksOffset(unsigned char masksOffset)
    {
        m_MasksOffset = masksOffset;
    }
    
    void setCalibVarsDirs(vector<string> calibVarsDirs)
    {
        m_CalibVarsDirs = calibVarsDirs;
    }
    
    void setFramesIndices(vector<string> framesIndices)
    {
        m_FramesIndices = framesIndices;
    }
    
    void setSceneLimits(vector< pair<int, int> > sceneLimits)
    {
        m_SceneLimits = sceneLimits;
    }
    
    
private:
    // Load data from disk: frames, masks, and rectangular bounding boxes
    vector<cv::Mat> m_Frames;

    //vector<cv::Mat> m_Masks; // groundtruth masks
    vector<cv::Mat> m_PredictedMasks; // bs predicted masks
    //vector<int> m_FrameIDs;
    //vector<cv::Mat> m_Masks;
    vector<cv::Mat> m_GroundTruthMasks;
    
    vector<string> m_FramesIndices;
    
    vector< vector<cv::Rect> > m_PredictedBoundingRects;
    vector< vector<cv::Rect> > m_GroundTruthBoundingRects;
    vector< vector<int> > m_Tags;
    vector<string> m_CalibVarsDirs;
    
    cv::Mat m_FramesResolutions;
    
    vector< pair<int, int> > m_SceneLimits;
    
    unsigned char m_MasksOffset;
};

#endif /* defined(__segmenthreetion__ModalityData__) */
