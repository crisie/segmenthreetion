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
    GridMat(unsigned int hp, unsigned int wp, unsigned int helems = 1, unsigned int welems = 1, int type = CV_32SC1);
    GridMat(cv::Mat mat, unsigned int hp = 2, unsigned int wp = 2);
    GridMat(GridMat& other, cv::Mat indices);
    //GridMat(GridMat& other);

    void create(unsigned int crows, unsigned int ccols);
    void copyTo(cv::Mat mat, unsigned int i, unsigned int j);
    
    // Get the grid cell matrix at (i,j)
    cv::Mat at(unsigned int i, unsigned int j) const;
    cv::Mat & get(unsigned int i, unsigned int j);
    
    unsigned int ccols() const;
    unsigned int crows() const;
    cv::Mat cols();
    cv::Mat rows();

    void hconcat(GridMat other);
    void vconcat(GridMat other);
    void hconcat(cv::Mat mat, unsigned int i, unsigned int j);
    void vconcat(cv::Mat mat, unsigned int i, unsigned int j);
    
    void set(cv::Mat cell, unsigned int i, unsigned int j);
    
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
