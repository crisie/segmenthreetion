//
//  ModalityWriter.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 10/03/14.
//
//

#ifndef __segmenthreetion__ModalityWriter__
#define __segmenthreetion__ModalityWriter__

#include <iostream>

#include "ModalityData.hpp"


class ModalityWriter
{
    
public:
    ModalityWriter();
    
    ModalityWriter(string dataPath);
    
    void setDataPath(string dataPath);
    
    void write(string modality, ModalityData& md);
    
    void saveValidBoundingBoxes(string file, vector<vector<int> > validBbs);
    
    void loadValidBoundingBoxes(string file, vector<vector<int> >& validBbs);
    
    void loadValidBoundingBoxes(string file, vector<int> & validBbs);
    
private:
    string m_DataPath;
    std::vector<std::string> m_ScenesPaths;
    
    // Save Mats in folder
    void saveMats(string dir, const char *format, vector<cv::Mat> frames, vector<string> filenames);
    // Save bounding rects and tags in .yml format
    void saveBoundingRects(string file, vector< vector<cv::Rect> > rects, vector< vector<int> > tags);
    
    void boundRectsToInt(vector<vector<cv::Rect> > bbModal, vector<vector<int> >& bbModalInt);
    
    template<typename T>
    void serialize(vector<vector<T> > m, vector<T>& s);
};


#endif /* defined(__segmenthreetion__ModalityWriter__) */
