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

template<typename T>
class GridMat
{
    friend ostream& operator<<( ostream&, const GridMat& );
    
public:
    /*
     * Constructors
     */
    GridMat(unsigned int hp = 2, unsigned int wp = 2);
    GridMat(unsigned int hp, unsigned int wp, unsigned int helems, unsigned int welems);
    GridMat(cv::Mat mat, unsigned int hp = 2, unsigned int wp = 2);
    GridMat(GridMat<T>& other, cv::Mat indices);

    void create(unsigned int hp = 2, unsigned int wp = 2);
    void create(unsigned int hp, unsigned int wp, unsigned int helems = 1, unsigned int welems = 1);
    
    void copyTo(cv::Mat mat, unsigned int i, unsigned int j);
    
    // Get the grid cell matrix at (i,j)
    cv::Mat& at(unsigned int i, unsigned int j);
    T& at(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
    
    cv::Mat get(unsigned int i, unsigned int j);
    
    unsigned int ccols() const;
    unsigned int crows() const;
    unsigned int cols(unsigned int i, unsigned int j);
    unsigned int rows(unsigned int i, unsigned int j);

    void hconcat(GridMat<T>& other);
    void vconcat(GridMat<T>& other);
    void hconcat(cv::Mat mat, unsigned int i, unsigned int j);
    void vconcat(cv::Mat mat, unsigned int i, unsigned int j);
    
    void set(cv::Mat cell, unsigned int i, unsigned int j);
    
    void mean(GridMat<T>& gmean, int dim = 0);
    void max(GridMat<T>& gmax, int dim = 0);
    void min(GridMat<T>& gmin, int dim = 0);
    void sum(GridMat<T>& gsum, int dim = 0);
    
    void argmax(GridMat<T>& gargmax);
    void argmin(GridMat<T>& gargmin);
    
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
    
    void init(GridMat & gridMat);
    bool isEmpty();
    
    bool accessible(unsigned int i, unsigned int j) const;    
};


#endif /* defined(__Segmenthreetion__GridMat__) */
