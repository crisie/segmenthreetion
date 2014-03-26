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

class ModalityReader
{
public:
    ModalityReader();
    ModalityReader(string dataPath);
    
    void setDataPath(string dataPath);
    void setMasksOffset(unsigned char offset);
    
    void read(string modality, ModalityData& md);
    // TODO:
    // void read(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, ModalityData& md);
	// void read(string modality, vector<string> sceneDirs, const char* filetype, ModalityData& md);
    
    // Read and grid all the data
    void read(string modality, string sceneDir, const char* filetype, int hp, int wp, ModalityGridData& mgd);
	void read(string modality, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    void read(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    
    // Read and grid all some data (omit frames and masks)
    void mockread(string modality, string sceneDir, const char* filetype, int hp, int wp, ModalityGridData& mgd);
	void mockread(string modality, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    void mockread(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    
private:
    string m_DataPath;
    vector<string> m_ScenesPaths;
    unsigned char m_MasksOffset;
    
    // TODO:
    // void read(string modality, string scenePath, const char* filetype, ModalityData& md);
    void readScene(string modality, string scenePath, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    void mockreadScene(string modality, string scenePath, const char* filetype, int hp, int wp, ModalityGridData& mgd);
    
	void loadFilenames(string dir, const char* fileExtension, vector<string>& filenames);
    // Load frames of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames);
    // Load frames and frames' indices of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames, vector<string>& indices);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector<vector<cv::Rect> >& rects, vector< vector<int> >& tags);
    // Save calibVars files directories
    void loadCalibVarsDir(string dir, vector<string>& calibVarsDirs);
};

#endif /* defined(__segmenthreetion__ModalityReader__) */
