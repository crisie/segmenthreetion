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
    GridMat(unsigned int hp = 0, unsigned int wp = 0, int type = 0);
    GridMat(cv::Mat mat, unsigned int hp, unsigned int wp, int type = 0);
    //GridMat(GridMat& other);

    void create(unsigned int crows, unsigned int ccols, int type = 0);
    void copyTo(cv::Mat mat, unsigned int i, unsigned int j);
    
    // Get the grid cell matrix at (i,j)
    cv::Mat at(unsigned int i, unsigned int j) const;
    cv::Mat & get(unsigned int i, unsigned int j);
    
    unsigned int ccols() const;
    unsigned int crows() const;
    cv::Mat cols();
    cv::Mat rows();

    int type();
    
    void hconcat(GridMat other);
    void vconcat(GridMat other);
    void hconcat(cv::Mat mat, unsigned int i, unsigned int j);
    void vconcat(cv::Mat mat, unsigned int i, unsigned int j);
    
    void set(cv::Mat cell, unsigned int i, unsigned int j);
    
	void saveFS(const string & filename);
    void show(const char* namedWindow);

    void release();
    
    enum { UNKNOWN = -1, OBJECT = 0, SUBJECT = 1 };

private:
    /*
     * Class attributes
     */
    vector<cv::Mat>     m_grid;
    unsigned int    m_crows; // Num of cell rows
    unsigned int    m_ccols; // Num of cell cols
    unsigned int    m_rows;
    unsigned int    m_cols;
    
    int             m_type; // 0: no type, 1: subject, 2: object
    
    
    void init(GridMat & gridMat);
    bool isEmpty();
    
    bool accessible(unsigned int i, unsigned int j) const;    
};


#endif /* defined(__Segmenthreetion__GridMat__) */
