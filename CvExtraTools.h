//
//  CvExtraTools.h
//  segmenthreetion
//
//  Created by Albert Clapés on 02/04/14.
//
//

#ifndef __segmenthreetion__CvExtraTools__
#define __segmenthreetion__CvExtraTools__

#include <iostream>
#include <string>

namespace cvx {
    void setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    void copyMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void copyMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void copyMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true);
    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    void load(std::string file, cv::Mat& mat, int format = cv::FileStorage::FORMAT_YAML);
    void save(std::string file, cv::Mat mat, int format = cv::FileStorage::FORMAT_YAML);
}

#endif /* defined(__segmenthreetion__CvExtraTools__) */
