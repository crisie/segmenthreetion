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

#include <opencv2/opencv.hpp>

namespace cvx {
    
    // There are not functions in OpenCV letting you select a set of arbitrary
    // spare rows or columns and create from them a new and smaller cv::Mat
    
    // src < dst, indices index dst
    void setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    // src > dst, indices index src
    void copyMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void copyMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void copyMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true);
    //    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true); // alternative implementation to copyMat
    //    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    //    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    //    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    void hmean(cv::Mat src, cv::Mat& mean); // row-wise mean
    void vmean(cv::Mat src, cv::Mat& mean); // column-wise mean
    
    // Data-to-disk and disk-to-data nice functions (hide cv::FileStorage declarations)
    void load(std::string file, cv::Mat& mat, int format = cv::FileStorage::FORMAT_YAML);
    void save(std::string file, cv::Mat mat, int format = cv::FileStorage::FORMAT_YAML);
}

#endif /* defined(__segmenthreetion__CvExtraTools__) */
