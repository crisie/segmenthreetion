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
    
    ModalityGridData(ModalityGridData& other, cv::Mat logicals)
    {
        m_hp = other.m_hp;
        m_wp = other.m_wp;
        
        for (int i = 0; i < other.getTags().size(); i++)
        {
            unsigned char logical = (logicals.rows > 1) ? logicals.at<int>(i,0) : logicals.at<int>(0,i);
            if (logical)
            {
                if (!other.isMock())
                {
                    addGridFrame(other.getGridFrame(i));
                    addGridMask(other.getGridMask(i));
                }
                addGridFrameID(other.getGridFrameID(i));
                addFramePath(other.getFramePath(i));
                addFrameFilename(other.getFrameFilename(i));
                addFrameResolution(other.getFrameResolution(i));
                addGridBoundingRect(other.getGridBoundingRect(i));
                addTag(other.getTag(i));
                addValidnesses(other.getValidnesses(i));
                // TODO: index descriptors
            }
        }
    }

	void clear()
	{
		m_GFrames.clear();
		m_GMasks.clear();
        m_MasksOffsets.clear();
		m_FrameIDs.clear();
        m_FramesPaths.clear();
        m_FramesFilenames.clear();
		m_FramesResolutions.clear();
		m_GBoundingRects.clear();
		m_Tags.clear();
        m_Validnesses.release();
        m_Descriptors.release();
	}
    
    void operator=(const ModalityGridData& other)
    {
        m_hp = other.m_hp;
        m_wp = other.m_wp;
        m_GFrames = other.m_GFrames;
        m_GMasks = other.m_GMasks;
        m_MasksOffsets = other.m_MasksOffsets;
        m_FrameIDs = other.m_FrameIDs;
        m_FramesPaths = other.m_FramesPaths;
        m_FramesFilenames = other.m_FramesFilenames;
        m_FramesResolutions = other.m_FramesResolutions;
        m_GBoundingRects = other.m_GBoundingRects;
        m_Tags = other.m_Tags;
        m_Validnesses = other.m_Validnesses;
        m_Descriptors = other.m_Descriptors;
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
    
    unsigned char getMaskOffset(int k)
    {
        return m_MasksOffsets[k];
    }
    
    int getGridFrameID(int k)
    {
        return m_FrameIDs[k];
    }
    
    string getFramePath(int k)
    {
        return m_FramesPaths[k];
    }
    
    string getFrameFilename(int k)
    {
        return m_FramesFilenames[k];
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
    
    cv::Mat getValidnesses(int k)
    {
        cv::Mat validness (m_hp, m_wp, cv::DataType<unsigned char>::type);
        
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            validness.at<unsigned char>(i,j) = m_Validnesses.at<unsigned char>(i,j,k,0);
        }
        
        return validness;
    }
    
    GridMat getDescriptor(int k)
    {
        GridMat gDescriptors (m_hp, m_wp);
        
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            gDescriptors.assign(m_Descriptors.at(i,j).row(k), i, j);
        }
        
        return gDescriptors;
    }
    
    cv::Mat& getDescriptors(unsigned int i, unsigned int j)
    {
        return m_Descriptors.at(i,j);
    }
    
    cv::Mat getDescriptors(unsigned int i, unsigned int j, int k)
    {
        return m_Descriptors.at(i,j).row(k);
    }
    
    GridMat getValidDescriptors()
    {
        return GridMat(m_Descriptors, m_Validnesses);
    }

    int getNumOfScenes()
    {
        return m_ScenesPaths.size();
    }
    
    vector<GridMat>& getGridsFrames()
    {
        return m_GFrames;
    }
    
    vector<GridMat>& getGridsMasks()
    {
        return m_GMasks;
    }
    
    vector<int>& getSceneIDs()
    {
        return m_SceneIDs;
    }
    
    vector<int>& getFrameIDs()
    {
        return m_FrameIDs;
    }
    
    vector<string>& getFramesPaths()
    {
        return m_FramesPaths;
    }
    
    vector<string>& getFramesFilenames()
    {
        return m_FramesFilenames;
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
    
    GridMat& getValidnesses()
    {
        return m_Validnesses;
    }
    
    cv::Mat& getValidnesses(unsigned int i, unsigned int j)
    {
        return m_Validnesses.at(i,j);
    }
    
    GridMat& getDescriptors()
    {
        return m_Descriptors;
    }

    cv::Mat getSceneIDsMat()
    {
		return cv::Mat(m_SceneIDs.size(), 1, cv::DataType<int>::type, m_SceneIDs.data());
    }
    
	cv::Mat getFrameIDsMat()
    {
		return cv::Mat(m_FrameIDs.size(), 1, cv::DataType<int>::type, m_FrameIDs.data());
    }
    
	cv::Mat getTagsMat()
    {
		return cv::Mat(m_Tags.size(), 1, cv::DataType<int>::type, m_Tags.data());
    }
    
    GridMat getValidTags()
    {
        GridMat gValidTags (m_hp, m_wp);
        
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            for (int k = 0; k < getValidnesses(i,j).rows; k++)
            {
                if (m_Validnesses.at<unsigned char>(i,j,k,0))
                    gValidTags.at(i,j).push_back(m_Tags[k]);
            }
        }
        
        return gValidTags;
    }
    
    int getHp()
    {
        return m_hp;
    }
    
    int getWp()
    {
        return m_wp;
    }
    
    bool isMock()
    {
        return m_GFrames.size() == 0 && m_GMasks.size() == 0 && m_Tags.size() > 0;
    }
    
    bool isDescribed()
    {
        return !m_Descriptors.isEmpty();
    }
    
    // Setters
    
    void setHp(int hp)
    {
        m_hp = hp;
    }
    
    void setWp(int wp)
    {
        m_wp = wp;
    }
    
    void setModality(string name)
    {
        m_ModalityName = name;
    }
    
    string getModality()
    {
        return m_ModalityName;
    }
    
    void setGridsFrames(vector<GridMat> gframes)
    {
        m_GFrames = gframes;
    }
    
    void setGridsMasks(vector<GridMat> gmasks)
    {
        m_GMasks = gmasks;
    }
    
    void setMasksOffsets(vector<unsigned char> gmasksoffsets)
    {
        m_MasksOffsets = gmasksoffsets;
    }
    
    void setGridsFrameIDs(vector<int> gframeids)
    {
        m_FrameIDs = gframeids;
    }
    
    void setFramesPaths(vector<string> paths)
    {
        m_FramesPaths = paths;
    }
    
    void setFramesFilenames(vector<string> filenames)
    {
        m_FramesFilenames = filenames;
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
    
    void setValidnesses(GridMat validnesses)
    {
        m_Validnesses = validnesses;
    }
    
    void setDescriptors(GridMat descriptors)
    {
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            int c = 0;
            for (int k = 0; k < m_Validnesses.at(i,j).rows; k++)
            {
                unsigned char bValidMask = m_Validnesses.at<unsigned char>(i,j,k,0);
                if (bValidMask) // nonzero pixels in mask
                {
                    cv::Mat g = descriptors.at(i,j);
                    cv::Mat d = g.row(c);
                    
                    bool bValidDescriptor = cv::checkRange(d);
                    
                    if (bValidDescriptor) m_Descriptors.at(i,j).push_back(d);
                    else m_Validnesses.at<unsigned char>(i,j,k,0) = 0;
                    
                    c++; // there are so many descriptors to check as 1s in validness (valid masks)
                }
            }
        }
    }
    
    void setDescriptors(cv::Mat descriptors, unsigned int i, unsigned int j)
    {
        m_Descriptors.at(i,j) = descriptors;
    }
    
    void setValidnesses(int k, cv::Mat validness)
    {
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            m_Validnesses.at<unsigned char>(i,j,k,0) = validness.at<unsigned char>(i,j);
        }
    }
    
    void setValidness(bool validness, unsigned int i, unsigned int j, unsigned int k)
    {
        m_Validnesses.at<unsigned char>(i,j,k,0) = validness ? 255 : 0;
    }

    void addGridFrame(GridMat gframe)
    {
        m_GFrames.push_back(gframe);
    }
    
    void addGridMask(GridMat gmask)
    {
		m_GMasks.push_back(gmask);
    }
    
    void addGridMaskOffset(unsigned char offset)
    {
        m_MasksOffsets.push_back(offset);
    }
    
    void addSceneID(int id)
    {
        m_SceneIDs.push_back(id);
    }
    
    void addScenePath(string scenePath)
    {
        m_ScenesPaths.push_back(scenePath);
    }
    
    void addGridFrameID(int id)
    {
        m_FrameIDs.push_back(id);
    }
    
    void addFramePath(string path)
    {
        m_FramesPaths.push_back(path);
    }
    
    void addFrameFilename(string filename)
    {
        m_FramesFilenames.push_back(filename);
    }
    
    void addFrameResolution(int x, int y)
    {
        m_FramesResolutions.push_back(cv::Point2d(x,y));
    }
    
    void addFrameResolution(cv::Point2d res)
    {
        m_FramesResolutions.push_back(res);
    }
    
    void addGridBoundingRect(cv::Rect gboundingrect)
    {
        m_GBoundingRects.push_back(gboundingrect);
    }
    
    void addTag(int tag)
    {
        m_Tags.push_back(tag);
    }
    
    void addValidnesses(cv::Mat validnesses)
    {
        GridMat g (validnesses, m_hp, m_wp);
        m_Validnesses.vconcat(g);
    }
    
    void addDescriptor(cv::Mat descriptor, unsigned int i, unsigned int j)
    {
        setValidness(cv::checkRange(descriptor), i, j, m_Descriptors.at(i,j).size());
        
        m_Descriptors.at(i,j).push_back(descriptor);
    }
    
    void saveDescription(string dataPath, string sequencePath, string filename)
    {
        m_Descriptors.save(dataPath + sequencePath + "Description/" + filename);
    }
    
    void saveDescription(string dataPath, vector<string> sequencesPaths, string filename)
    {
        for (int i = 0; i < sequencesPaths.size(); i++)
            saveDescription(dataPath, sequencesPaths[i], filename);
    }
    
    void loadDescription(string dataPath, string sequencePath, string filename)
    {
        GridMat descriptors;
        descriptors.load(dataPath + sequencePath + "Description/" + filename);
        
        setDescriptors(descriptors);
    }
    
    void loadDescription(string dataPath, vector<string> sequencesPaths, string filename)
    {
        GridMat descriptors;
        
        for (int i = 0; i < sequencesPaths.size(); i++)
        {
            GridMat aux;
            aux.load(dataPath + sequencesPaths[i] + "Description/" + filename);
            descriptors.vconcat(aux);
        }
        
        setDescriptors(descriptors);
    }


private:
    int m_hp, m_wp;
    string m_ModalityName;
    
    vector<GridMat> m_GFrames;
    vector<GridMat> m_GMasks;
    vector<unsigned char> m_MasksOffsets;
    vector<int> m_FrameIDs;
    vector<int> m_SceneIDs;
    vector<string> m_ScenesPaths;
    vector<string> m_FramesPaths;
    vector<string> m_FramesFilenames;
    vector<cv::Point2d> m_FramesResolutions;
    vector<cv::Rect> m_GBoundingRects;
    vector<int> m_Tags;
    
    GridMat m_Validnesses; // whether cells in the grids are valid to be described
    GridMat m_Descriptors;
};


#endif /* defined(__segmenthreetion__ModalityGridData__) */
