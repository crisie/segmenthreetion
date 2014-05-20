//
//  ModalityReader.h
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#ifndef __segmenthreetion__ModalityReader__
#define __segmenthreetion__ModalityReader__

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ModalityData.hpp"
#include "ModalityGridData.hpp"

#include "CvExtraTools.h"

class ModalityReader
{
public:
    ModalityReader();
    ModalityReader(string dataPath);
    
    unsigned int getNumOfScenes();
    string getScenePath(unsigned int sid);
    
    void setDataPath(string dataPath);
    void setMasksOffset(unsigned char offset);
    
    cv::Mat getScenePartition(unsigned int sid);
    vector<cv::Mat> getPartitions();
    cv::Mat getAllScenesPartition();
    
    void read(string modality, ModalityData& md);
    // TODO:
    // void read(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, ModalityData& md);
	// void read(string modality, vector<string> sceneDirs, const char* filetype, ModalityData& md);
    
    // Read and grid all the data
	void readAllScenesData(string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    void readSceneData(unsigned int sid, string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd);

    // Read and grid all some data (omit frames and masks)
	void readAllScenesMetadata(string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    void readSceneMetadata(unsigned int sid, string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd);

    
    //Read only predicted and gt mask for computing overlap
    void overlapreadScene(string predictionType, string modality, string scenePath, const char* filetype, ModalityData& md);
    void overlapreadScene(string predictionType, string modality, string dataPath, string scenePath, const char* filetype, ModalityData& md);
    
    void agreement(vector<ModalityGridData*> mgds);
    
    void loadDescription(string modality, ModalityGridData& mgd);
    
    void getBoundingBoxesFromGroundtruthMasks(string modality, vector<string> sceneDirs, vector<vector<cv::Rect> >& boxes);
    void getBoundingBoxesFromGroundtruthMasks(string modality, string sceneDir, vector<vector<cv::Rect> >& boxes);
    
private:
    string m_DataPath;
    vector<string> m_ScenesPaths;
    unsigned char m_MasksOffset;
    unsigned char m_MaxOffset;
    
    double m_MinVal, m_MaxVal;

    
	void loadFilenames(string dir, const char* fileExtension, vector<string>& filenames);
    // Load frames of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames);
    // Load frames and frames' indices of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames, vector<string>& indices);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector<vector<cv::Rect> >& rects, vector< vector<int> >& tags);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector<vector<cv::Rect> >& rects);
    // Save calibVars files directories
    void loadCalibVarsDir(string dir, vector<string>& calibVarsDirs);
    
    void getBoundingBoxesInMask(cv::Mat mask, vector<cv::Rect>& boxes);
    
    void addScenePartition(cv::Mat partition);
};

#endif /* defined(__segmenthreetion__ModalityReader__) */
