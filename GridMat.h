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
    GridMat(GridMat& other, cv::Mat indices, int dim = 0, bool logical = true);

    void create(unsigned int hp = 2, unsigned int wp = 2);
    
    template<typename T>
    void create(unsigned int hp, unsigned int wp, unsigned int helems = 1, unsigned int welems = 1);
    
    void copyTo(cv::Mat mat, unsigned int i, unsigned int j);
    
    // Get the grid cell matrix at (i,j)
    cv::Mat& at(unsigned int i, unsigned int j);
    
    template<typename T>
    T& at(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
    
    cv::Mat get(unsigned int i, unsigned int j) const;
    void set(cv::Mat cell, unsigned int i, unsigned int j);
    
    GridMat vget(cv::Mat indices, bool logical = true);
    GridMat hget(cv::Mat indices, bool logical = true);
    
    void vset(GridMat grid, cv::Mat indices, bool logical = true);
    void hset(GridMat grid, cv::Mat indices, bool logical = true);
    
    unsigned int ccols() const;
    unsigned int crows() const;
    unsigned int cols(unsigned int i, unsigned int j);
    unsigned int rows(unsigned int i, unsigned int j);

    void hconcat(GridMat& other);
    void vconcat(GridMat& other);
    void hconcat(cv::Mat mat, unsigned int i, unsigned int j);
    void vconcat(cv::Mat mat, unsigned int i, unsigned int j);
    
    void mean(GridMat& gmean, int dim = 0);
    void max(GridMat& gmax, int dim = 0);
    void min(GridMat& gmin, int dim = 0);
    void sum(GridMat& gsum, int dim = 0);
    
    template<typename T>
    void argmax(GridMat& gargmax);
    template<typename T>
    void argmin(GridMat& gargmax);
    
	void saveFS(const string & filename);
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
    bool isEmpty();
    
    bool accessible(unsigned int i, unsigned int j) const;
    
    GridMat getIndexedCellElements(cv::Mat indices, int dim = 0, bool logical = true);
    void setIndexedCellElements(GridMat grid, cv::Mat indices, int dim = 0, bool logical = true);
    
    GridMat getIndexedCellElementsLogically(cv::Mat indices, int dim = 0);
    void setIndexedCellElementsLogically(GridMat grid, cv::Mat indices, int dim = 0);
    
    GridMat getIndexedCellElementsPositionally(cv::Mat indices, int dim = 0);
    void setIndexedCellElementsPositionally(GridMat grid, cv::Mat indices, int dim = 0);
};


#endif /* defined(__Segmenthreetion__GridMat__) */
