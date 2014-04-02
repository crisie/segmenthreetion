//
//  CvExtraTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 02/04/14.
//
//

#include "CvExtraTools.h"

void cvx::setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatLogically(src, dst, indices);
    else
        setMatPositionally(src, dst, indices);
}

void cvx::setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    cv::Mat aux = src;
    bool rowwise = logicals.rows > 1;
    
    if (dst.empty())
    {
        if (rowwise)
            dst.create(logicals.rows, src.cols, src.type());
        else
            dst.create(src.rows, logicals.rows, src.type());
    }
    
    int c = 0;
    int n = rowwise ? dst.rows : dst.cols;
    for (int i = 0; i < n; i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? src.row(c++).copyTo(dst.row(i)) : src.col(c++).copyTo(dst.col(i));
    }
}

void cvx::setMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    bool rowwise = indexes.rows > 1;
    
    int n = rowwise ? indexes.rows : indexes.cols;
    for (int i = 0; i < n; i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(i).copyTo(dst.row(idx)) : src.col(i).copyTo(dst.col(idx));
    }
}

void cvx::copyMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatLogically(src, dst, indices);
    else
        setMatPositionally(src, dst, indices);
}

void cvx::copyMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = logicals.rows > 1;
    
    int c = 0;
    for (int i = 0; i < rowwise ? logicals.rows : logicals.cols; i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? src.row(i).copyTo(dst.row(c++)) : src.col(i).copyTo(dst.col(c++));
    }
}

void cvx::copyMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = indexes.rows > 1;
    
    for (int i = 0; i < rowwise ? indexes.rows : indexes.cols; i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(idx).copyTo(dst.row(i)) : src.col(idx).copyTo(dst.col(i));
    }
}

cv::Mat cvx::indexMat(cv::Mat src, cv::Mat indices, bool logical)
{
    cv::Mat mat;
    
    cvx::copyMat(src, indices, logical);

    return mat;
}

//void cvx::indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
//{
//    if (logical)
//        indexMatLogically(src, dst, indices);
//    else
//        indexMatPositionally(src, dst, indices);
//}
//
//void cvx::indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
//{
//    if (logicals.rows > 1) // row-wise
//        dst.create(0, src.cols, src.type());
//    else // col-wise
//        dst.create(src.rows, 0, src.type());
//
//    for (int i = 0; i < (logicals.rows > 1 ? logicals.rows : logicals.cols); i++)
//    {
//        if (logicals.rows > 1)
//        {
//            if (logicals.at<unsigned char>(i,0))
//            {
//                dst.push_back(src.row(i));
//            }
//        }
//        else
//        {
//            if (logicals.at<unsigned char>(0,i))
//                dst.push_back(src.col(i));
//        }
//    }
//}
//
//void cvx::indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices)
//{
//    dst.release();
//
//    if (indices.rows > 1) // row-wise
//        dst.create(indices.rows, src.cols, src.type());
//    else // col-wise
//        dst.create(src.rows, indices.cols, src.type());
//
//    for (int i = 0; i < (indices.rows > 1 ? indices.rows : indices.cols); i++)
//    {
//        int idx = indices.rows > 1 ? indices.at<int>(i,0) : indices.at<int>(0,i);
//
//        if (indices.rows > 1)
//            src.row(idx).copyTo(dst.row(i));
//        else
//            src.col(idx).copyTo(dst.col(i));
//    }
//}

void cvx::hmean(cv::Mat src, cv::Mat& mean)
{
    mean.release();
    mean.create(src.rows, 1, cv::DataType<double>::type);
    
    for (int i = 0; src.rows; i++)
    {
        mean.at<double>(i,0) = cv::mean(src.row(i)).val[0];
    }
}

void cvx::vmean(cv::Mat src, cv::Mat& mean)
{
    mean.release();
    mean.create(1, src.cols, cv::DataType<double>::type);
    
    for (int i = 0; src.cols; i++)
    {
        mean.at<double>(0,i) = cv::mean(src.col(i)).val[0];
    }
}


void cvx::load(std::string file, cv::Mat& mat, int format)
{
    cv::FileStorage fs(file, FileStorage::READ | format);
    
    fs["mat"] >> mat;
    
    fs.release();
    
}

void cvx::save(std::string file, cv::Mat mat, int format)
{
    cv::FileStorage fs(file, FileStorage::WRITE | format);
    
    fs << "mat" << mat;
    
    fs.release();
    
}