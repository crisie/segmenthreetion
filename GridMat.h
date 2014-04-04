//
//  GridMat.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 21/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__GridMat__
#define __Segmenthreetion__GridMat__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;

class GridMat
{
    friend ostream& operator<<( ostream&, const GridMat& );
    
public:
    /*
     * Constructors
     */
    GridMat(unsigned int hp = 2, unsigned int wp = 2);
    GridMat(unsigned int hp, unsigned int wp, unsigned int helems, unsigned int welems, int type = CV_32FC1);
    GridMat(cv::Mat mat, unsigned int hp = 2, unsigned int wp = 2);
    GridMat(const GridMat& other);
    GridMat(GridMat& other, GridMat indices, int k, bool inverse = false);
    GridMat(GridMat& other, GridMat indices, bool logical = true);

    void create(unsigned int hp = 2, unsigned int wp = 2);
    
    template<typename T>
    void create(unsigned int hp, unsigned int wp, unsigned int helems = 1, unsigned int welems = 1);
    
    void copyTo(cv::Mat mat, unsigned int i, unsigned int j);
    
    bool isEmpty();
    
    // Get the grid cell matrix at (i,j)
    cv::Mat& at(unsigned int i, unsigned int j);
    
    template<typename T>
    T& at(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
    
    cv::Mat get(unsigned int i, unsigned int j) const;
    
    cv::Rect getCellCoordinates(unsigned int i, unsigned int j);
    
    void assign(cv::Mat cell, unsigned int i, unsigned int j);
    void set(GridMat& other);
    
    template<typename T>
    void setTo(T value);
    template<typename T>
    void setTo(T value, unsigned int i, unsigned int j);
    template<typename T>
    void setTo(T value, unsigned int i, unsigned int j, cv::Mat mask);
    //void setTo(cv::Mat& m);
    //void setTo(GridMat& m);
    
    template<typename T>
    void convertToMat(cv::Mat& mat);
    template<typename T>
    cv::Mat convertToMat();
    
    GridMat convertToSparse(GridMat indices);
    void convertToSparse(GridMat indices, GridMat& sparseGridMat);
    GridMat convertToDense(GridMat indices);
    void convertToDense(GridMat indices, GridMat& denseGridMat);
    
    void normalize(GridMat& g); // at cell level
    GridMat normalize(); // at cell level
    void standardize(GridMat& g);
    GridMat standardize();
    
    void set(GridMat src, GridMat indices); // size(src) < size(dst)
    void set(GridMat src, GridMat indices, int k); // size(src) < size(dst)
    void index(GridMat& dst, GridMat indices, int k); // size(src) > size(dst)
    void copyTo(GridMat& dst);
    void copyTo(GridMat& dst, GridMat indices); // size(src) == src(dst)
    void copyTo(GridMat& dst, GridMat indices, int k); // size(src) == size(dst)
    
//    GridMat operator<(GridMat& other);
//    GridMat operator<=(GridMat& other);
//    GridMat operator>(GridMat& other);
//    GridMat operator>=(GridMat& other);
//    GridMat operator==(GridMat& other);
//    GridMat operator!=(GridMat& other);
//    GridMat operator+(GridMat& other);
//    GridMat operator-(GridMat& other);
//    GridMat operator*(GridMat& other);
//    GridMat operator/(GridMat& other);
    
    template<typename T>
    GridMat operator<(T value);
    template<typename T>
    GridMat operator<=(T value);
    template<typename T>
    GridMat operator>(T value);
    template<typename T>
    GridMat operator>=(T value);
    template<typename T>
    GridMat operator==(T value);
    template<typename T>
    GridMat operator!=(T value);
    template<typename T>
    GridMat operator+(T value);
    template<typename T>
    GridMat operator-(T value);
    template<typename T>
    GridMat operator*(T value);
    template<typename T>
    GridMat operator/(T value);

//    GridMat vget(cv::Mat indices, bool logical = true);
//    GridMat hget(cv::Mat indices, bool logical = true);
//    
//    void vset(GridMat grid, cv::Mat indices, bool logical = true);
//    void hset(GridMat grid, cv::Mat indices, bool logical = true);
    
    unsigned int ccols() const;
    unsigned int crows() const;
    unsigned int cols(unsigned int i, unsigned int j);
    unsigned int rows(unsigned int i, unsigned int j);

    void hconcat(GridMat& other);
    void vconcat(GridMat& other);
    void hconcat(cv::Mat& mat, unsigned int i, unsigned int j);
    void vconcat(cv::Mat& mat, unsigned int i, unsigned int j);
    
    void hserial(cv::Mat& serial);
    void vserial(cv::Mat& serial);
    
    // Cells' collapse functions. Each cell into a row or a column.
    void mean(GridMat& gmean, int dim = 0);
    void max(GridMat& gmax, int dim = 0);
    void min(GridMat& gmin, int dim = 0);
    void sum(GridMat& gsum, int dim = 0);
    GridMat abs();
    
    // Grid's collapse functions. The grid is turnet into a single-grid (Mat)
    cv::Mat accumulate();
    void accumulate(cv::Mat& accumulation);
    cv::Mat average();
    void average(cv::Mat& average);
    void biaccumulate(GridMat g, int threshold, cv::Mat& acc1, cv::Mat& acc2);
    void biaverage(GridMat g, int threshold, cv::Mat& acc1, cv::Mat& acc2);
    
    template<typename T> cv::Mat findNonZero();
    
    void argmax(GridMat& gargmax);
    void argmin(GridMat& gargmax);
    
    GridMat historize (int nbins, double min, double max);
    
	void save(const string & filename);
    void load(const string & filename);
    void show(const char* namedWindow);

    void release();

private:
    /*
     * Class attributes
     */
    vector<cv::Mat>     m_grid;
    unsigned int    m_crows; // Num of cell rows
    unsigned int    m_ccols; // Num of cell cols
    unsigned int    m_rows;
    unsigned int    m_cols;
    
    void init(GridMat& other);

    bool accessible(unsigned int i, unsigned int j) const;
    
//    GridMat getIndexedCellElements(cv::Mat indices, int dim = 0, bool logical = true);
//    void setIndexedCellElements(GridMat& grid, cv::Mat indices, int dim = 0, bool logical = true);
//    
//    GridMat getIndexedCellElementsLogically(cv::Mat logicals, int dim = 0);
//    void setIndexedCellElementsLogically(GridMat& grid, cv::Mat logicals, int dim = 0);
//    
//    GridMat getIndexedCellElementsPositionally(cv::Mat indices, int dim = 0);
//    void setIndexedCellElementsPositionally(GridMat& grid, cv::Mat indices, int dim = 0);
};

#endif /* defined(__Segmenthreetion__GridMat__) */
