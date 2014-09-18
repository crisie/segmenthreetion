//
//  BackgroundSubtractor.h
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 05/03/14.
//
//

#ifndef __segmenthreetion__BackgroundSubtractor__
#define __segmenthreetion__BackgroundSubtractor__

#include <iostream>

#include <boost/algorithm/string.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ModalityData.hpp"


using namespace std;


class BackgroundSubtractor {
    
private:
    
    unsigned char m_masksOffset;
    
    enum boolPerson {
		FALSE = 0,
		TRUE = 1,
		UNDEFINED = -1
	};
    
    cv::Point getTopLeftPoint(cv::Point p1, cv::Point p2);
    
    cv::Point getBottomRightPoint(cv::Point p1, cv::Point p2);
    
public:
    
    BackgroundSubtractor();
    
    void setMasksOffset(unsigned char masksOffset);
    
    unsigned char getMasksOffset();
    
    void getGroundTruthBoundingRects(ModalityData& md);
    
    void getRoiTags(ModalityData& md, bool manualAid);
    
protected:
    void getMaximalBoundingBox(vector<cv::Rect> &boundingBox, cv::Size limits, cv::Rect & outputBoundingBox);
    
    void changePixelValue(cv::Mat & mask, int pixelValue);
    
    void changePixelValue(cv::Mat & mask, int threshold, int pixelValue);
    
    void checkWhitePixels(cv::Rect & box, cv::Mat frame);

    int countBoundingBoxes(vector<vector<cv::Rect> > boundingBoxes);
    
    void getMaskBoundingBoxes(cv::Mat mask, vector<cv::Rect> & boundingBoxes);

    bool checkMinimumBoundingBoxes(cv::Rect box, int min);
    
    cv::Rect getMinimumBoundingBox(cv::Rect box, int min);
    
    int isPersonBox(cv::Rect r1, cv::Rect r2);
    
};

#endif /* defined(__segmenthreetion__BackgroundSubtractor__) */




