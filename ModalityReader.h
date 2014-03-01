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

class ModalityReader
{
public:
    ModalityReader();
    ModalityReader(std::string dataPath);
    
    void setDataPath(std::string dataPath);
    void setMasksOffset(unsigned char offset);
    
    void read(std::string modality, ModalityData& md);
    
private:
    std::string m_DataPath;
    std::vector<std::string> m_ScenesPaths;
    unsigned char m_MasksOffset;
    
    // Load frames of a modality within a directory
    void loadDataToMats(std::string dir, const char* format, std::vector<cv::Mat>& frames);
    // Load people bounding boxes (rects)
    void loadBoundingRects(std::string file, std::vector< std::vector<cv::Rect> >& rects, std::vector< std::vector<int> >& tags);
};

#endif /* defined(__segmenthreetion__ModalityReader__) */
