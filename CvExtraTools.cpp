//
//  CvExtraTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 02/04/14.
//
//

#include "CvExtraTools.h"

#include <matio.h>

void cvx::setMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatLogically(src, dst, indices);
    else
        setMatPositionally(src, dst, indices);
}

void cvx::setMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
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

void cvx::indexMat(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        indexMatLogically(src, dst, indices);
    else
        indexMatPositionally(src, dst, indices);
}

void cvx::indexMatLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    bool rowwise = logicals.rows > 1;
    
    if (dst.empty())
    {
        if (rowwise)
            dst.create(0, src.cols, src.type());
        else
            dst.create(src.rows, 0, src.type());
    }
    
    int c = 0;
    for (int i = 0; i < (rowwise ? logicals.rows : logicals.cols); i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? dst.push_back(src.row(i)) : dst.push_back(src.col(i));
    }
}

void cvx::indexMatPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = indexes.rows > 1;
    
    for (int i = 0; i < (rowwise ? indexes.rows : indexes.cols); i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(idx).copyTo(dst.row(i)) : src.col(idx).copyTo(dst.col(i));
    }
}

cv::Mat cvx::indexMat(cv::Mat src, cv::Mat indices, bool logical)
{
    cv::Mat mat;
    
    cvx::indexMat(src, mat, indices, logical);

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

void cvx::hist(cv::Mat src, int nbins, float min, float max, cv::Mat& hist)
{
    int histSize[] = { nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float range[] = { min, max } ;
    const float* ranges[] = { range };
    
    src.convertTo(src, CV_32F);
    
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, true);
}

void cvx::hist(cv::Mat src, cv::Mat msk, int nbins, float min, float max, cv::Mat& hist)
{
    int histSize[] = { nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float range[] = { min, max } ;
    const float* ranges[] = { range };
    
    src.convertTo(src, CV_32F);
    
    cv::calcHist(&src, 1, channels, msk, hist, 1, histSize, ranges, true, true);
}

void cvx::cumsum(cv::Mat src, cv::Mat& dst)
{
    dst = src.clone();
    
    //cv::add(dst.row(0), src.row(0), dst.row(0));
    
    for (int i = 0; i < src.rows - 1; i++)
        for (int j = i+1; j < src.rows; j++)
            cv::add(dst.row(j), src.row(i), dst.row(j));
    
    for (int i = 0; i < src.cols - 1; i++)
        for (int j = i+1; j < src.cols; j++)
            cv::add(dst.col(j), src.col(j), dst.col(j));
}

cv::Mat cvx::linspace(int start, int end)
{
    cv::Mat l;
    
    cvx::linspace(start, end, l);
    
    return l;
}

cv::Mat cvx::linspace(float start, float end, int n)
{
    cv::Mat l;
    
    cvx::linspace(start, end, n, l);
    
    return l;
}

void cvx::linspace(int start, int end, cv::Mat& mat)
{
    std::vector<int> v;
    cvx::linspace(start, end, v);
    
    mat.create(v.size(), 1, cv::DataType<int>::type);
    memcpy(mat.data, v.data(), sizeof(int) * v.size());
}

void cvx::linspace(float start, float end, int n, cv::Mat& mat)
{
    std::vector<float> v;
    cvx::linspace(start, end, n, v);

    mat.create(v.size(), 1, cv::DataType<float>::type);
    memcpy(mat.data, v.data(), sizeof(float) * v.size());
}

void cvx::linspace(int start, int end, std::vector<int>& v)
{
    for (int i = start; i < end; i++)
    {
        v.push_back(i);
    }
}

void cvx::linspace(float start, float end, int n, std::vector<float>& v)
{
    for (int i = 0; i < n; i++)
    {
        v.push_back(start + i * (end - start) / (n - 1));
    }
}

void cvx::load(std::string file, cv::Mat& mat, int format)
{
    cv::FileStorage fs(file, cv::FileStorage::READ | format);
    
    fs["mat"] >> mat;
    
    fs.release();
    
}

void cvx::save(std::string file, cv::Mat mat, int format)
{
    cv::FileStorage fs(file, cv::FileStorage::WRITE | format);
    
    fs << "mat" << mat;
    
    fs.release();
    
}

template<typename T>
cv::Mat cvx::matlabread(std::string file)
{
    cv::Mat mat;
    
    cvx::matlabread<T>(file, mat);
    
    return mat;
}

template<typename T>
void cvx::matlabread(std::string file, cv::Mat& mat)
{
    mat_t *matfp;
    matvar_t *matvar;

    matfp = Mat_Open(file.c_str(), MAT_ACC_RDONLY);
    
    if ( NULL == matfp ) {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n", file.c_str());
        return;
    }

    matvar = Mat_VarRead(matfp,"perMap");
    if ( NULL == matvar ) {
        fprintf(stderr, "Variable ’perMap’ not found, or error "
                "reading MAT file\n");
    }
    else
    {
        int nrows, ncols;
        nrows = matvar->dims[0];
        ncols = matvar->dims[1];
        
        cv::Mat rmap (nrows, ncols, cv::DataType<T>::type, matvar->data);
        mat = rmap.clone();
        
        Mat_VarFree(matvar);
    }
    
    Mat_Close(matfp);
}

template cv::Mat cvx::matlabread<int>(std::string file);
template cv::Mat cvx::matlabread<float>(std::string file);
template cv::Mat cvx::matlabread<double>(std::string file);

template void cvx::matlabread<int>(std::string file, cv::Mat& mat);
template void cvx::matlabread<float>(std::string file, cv::Mat& mat);
template void cvx::matlabread<double>(std::string file, cv::Mat& mat);
