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
#include <vector>

#include <opencv2/opencv.hpp>

namespace cvx
{
    
    // There are not functions in OpenCV letting you select a set of arbitrary
    // spare rows or columns and create from them a new and smaller cv::Mat
    
    // src < dst, indices index dst
    void setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    // src > dst, indices index src
    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true);
    //    cv::Mat indexMat(cv::Mat src, cv::Mat indices, bool logical = true); // alternative implementation to copyMat
    //    void indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical = true);
    //    void indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals);
    //    void indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices);
    
    // Wrappers for cv::collapse(...)
    void hmean(cv::Mat src, cv::Mat& mean); // row-wise mean
    void vmean(cv::Mat src, cv::Mat& mean); // column-wise mean
    
    void hist(cv::Mat src, int nbins, float min, float max, cv::Mat& hist);
    void hist(cv::Mat src, cv::Mat msk, int nbins, float min, float max, cv::Mat& hist);
    
    void cumsum(cv::Mat src, cv::Mat& dst);
    
    void linspace(float start, float end, int n, cv::Mat& m);
    void linspace(float start, float end, int n, std::vector<float>& v);
    cv::Mat linspace(float start, float end, int n);
    
    void linspace(int start, int end, cv::Mat& m);
    void linspace(int start, int end, std::vector<int>& v);
    cv::Mat linspace(int start, int end);
    
    // Data-to-disk and disk-to-data nice functions (hide cv::FileStorage declarations)
    void load(std::string file, cv::Mat& mat, int format = cv::FileStorage::FORMAT_YAML);
    void save(std::string file, cv::Mat mat, int format = cv::FileStorage::FORMAT_YAML);
    
    template<typename T>
    cv::Mat matlabread(std::string file);
    template<typename T>
    void matlabread(std::string file, cv::Mat& mat);
    
    void computePCA(cv::Mat src, cv::PCA& pca, cv::Mat& dst, int flags = CV_PCA_DATA_AS_ROW, double variance = 1);
}


#endif /* defined(__segmenthreetion__CvExtraTools__) */
