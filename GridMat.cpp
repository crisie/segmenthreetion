//
//  GridMat.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 21/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "GridMat.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

template<typename T>
GridMat<T>::GridMat(unsigned int crows, unsigned int ccols) : m_crows(crows), m_ccols(ccols)
{    
    m_grid.resize( m_crows * m_ccols );
}

template<typename T>
GridMat<T>::GridMat(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems)
: m_crows(crows), m_ccols(ccols)
{
    m_grid.resize( m_crows * m_ccols );
    for (unsigned int row = 0; row < m_crows; row++) for (unsigned int col = 0; col < m_ccols; col++)
    {
        this->set(cv::Mat_<T>(helems, welems), row, col);
    }
}

template<typename T>
GridMat<T>::GridMat(cv::Mat_<T> mat, unsigned int crows, unsigned int ccols)
: m_crows(crows), m_ccols(ccols)
{
    m_rows = mat.rows;
    m_cols = mat.cols;
    
    m_grid.resize( m_crows * m_ccols );
    
    int a = m_rows/m_crows + (m_rows % m_crows)/m_crows;
    int b = m_cols/m_ccols + (m_cols % m_ccols)/m_ccols;
    unsigned int cellh, cellw;
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cellh = ((j+1) * b - 1) - (j * b);
        cellw = ((i+1) * a - 1) - (i * a);
        cv::Rect cell = cv::Rect((j * b), (i * a), cellh, cellw);
        cv::Mat_<T> roi(mat,cell);
        //copyMakeBorder(roi, roi, margin, margin, margin, margin, BORDER_CONSTANT, 0);
        this->set(roi,i,j);
    }
}

template<typename T>
GridMat<T>::GridMat(GridMat& other, cv::Mat_<unsigned int> indices)
{
    int nelems = cv::sum(indices).val[0];
    
    for (int i = 0; i < other.crows(); i++) for (int j = 0; j < other.crows(); j++)
    {
        cv::Mat& cell = other.get(i,j);
        cv::Mat cellPartition (nelems, cell.cols, cell.type());
        
        int n = 0;
        for (int d = 0; d < cell.cols; d++)
        {
            bool include = (indices.rows > 1) ? indices.at<int>(d,0) : indices.at<int>(0,d);
            if (include)
            {
                cell.row(d).copyTo(cellPartition.row(n++));
            }
        }
        
        set(cellPartition, i, j);
    }
}

/*
 * TODO: Implement this alternative instead of the nasty init method
 */
//GridMat::GridMat(GridMat & other)
//{
//    this->m_crows = other.crows();
//    this->m_ccols = other.ccols();
//    this->m_rows = other.m_rows;
//    this->m_cols = other.m_cols;
//    
//    this->m_grid.resize( m_crows * m_ccols );
//    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
//    {
//        this->set(other.at(i,j), i, j);
//    }
//}

template<typename T>
void GridMat<T>::create(unsigned int crows, unsigned int ccols)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
}


template<typename T>
void GridMat<T>::create(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        this->set(cv::Mat_<T>(helems, welems), i, j);
    }
}


template<typename T>
void GridMat<T>::init(GridMat & other)
{
    this->m_crows = other.crows();
    this->m_ccols = other.ccols();
    
    this->m_grid.resize( m_crows * m_ccols );
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        this->set(other.at(i,j), i, j);
    }
}

template<typename T>
void GridMat<T>::copyTo(cv::Mat_<T> mat, unsigned int i, unsigned int j)
{
    mat.copyTo(m_grid[i * ccols() + j]);
}

template<typename T>
unsigned int GridMat<T>::crows() const
{
//    std::cout << m_crows << std::endl;
    return m_crows;
}

template<typename T>
unsigned int GridMat<T>::ccols() const
{
//    std::cout << m_ccols << std::endl;
    return m_ccols;
}

template<typename T>
unsigned int GridMat<T>::rows(unsigned int i, unsigned int j)
{
    return m_grid[i * ccols() + j].rows;
}

template<typename T>
unsigned int GridMat<T>::cols(unsigned int i, unsigned int j)
{
    return m_grid[i * ccols() + j].cols;
}

template<typename T>
cv::Mat& GridMat<T>::at(unsigned int i, unsigned int j)
{
    return m_grid[i * m_ccols + j];
}

template<typename T>
T& GridMat<T>::at(unsigned int i, unsigned int j, unsigned int row, unsigned int col)
{
    return m_grid[i * m_ccols + j].template at<T>(row,col);
}

template<typename T>
cv::Mat GridMat<T>::get(unsigned int i, unsigned int j)
{
    return m_grid[i * m_ccols + j];
}

template<typename T>
void GridMat<T>::set(cv::Mat cell, unsigned int i, unsigned int j)
{
    this->at(i,j).release();
    m_grid[i * m_ccols + j] = cell;
}

template<typename T>
bool GridMat<T>::isEmpty()
{
    return ccols() == 0 || crows() == 0;
}

template<typename T>
void GridMat<T>::hconcat(GridMat<T>& other)
{
    // Set rather than concatenate if empty
    if (this->isEmpty())
    {
//        GridMat(other); // TODO
        this->init(other);
    }
    else
    {
        // Behave normally
        for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
        {
            cv::Mat concatenation;
            cv::hconcat(this->get(i,j), other.at(i,j), concatenation);
            this->at(i,j).release();
            this->set(concatenation, i,j);
        }
    }
}

template<typename T>
void GridMat<T>::vconcat(GridMat<T>& other)
{
    // Set rather than concatenate if empty
    if (this->isEmpty())
    {
        //        GridMat(other); // TODO
        this->init(other);
    }
    else
    {
        // Concatenate
        for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
        {
            cv::Mat concatenation;
            cv::vconcat(this->get(i,j), other.at(i,j), concatenation);
            this->at(i,j).release();
            this->set(concatenation, i,j);
        }
    }
}

template<typename T>
void GridMat<T>::hconcat(cv::Mat mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->set(mat,i,j);
    else
    {
        cv::hconcat(this->at(i,j), mat, this->at(i,j));
        mat.release();
    }
}

template<typename T>
void GridMat<T>::vconcat(cv::Mat mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->set(mat,i,j);
    else
    {
        cv::vconcat(this->at(i,j), mat, this->at(i,j));
        mat.release();
    }
}


template<typename T>
void GridMat<T>::mean(GridMat<T>& gmean, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat meanCol;
        cv::reduce(this->at(i,j), meanCol, dim, CV_REDUCE_AVG); // dim: 0 row-wise, 1 column-wise
        gmean.set(meanCol, i, j);
    }
}


template<typename T>
void GridMat<T>::max(GridMat<T>& gmax, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MAX); // dim: 0 row-wise, 1 column-wise
        gmax.set(col, i, j);
    }
}


template<typename T>
void GridMat<T>::min(GridMat<T>& gmin, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MIN); // dim: 0 row-wise, 1 column-wise
        gmin.set(col, i, j);
    }
}


template<typename T>
void GridMat<T>::sum(GridMat<T>&gsum, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_SUM); // dim: 0 row-wise, 1 column-wise
        gsum.set(col, i, j);
    }
}


template<typename T>
void GridMat<T>::argmax(GridMat<T>& gargmax)
{
    gargmax.create(m_crows, m_ccols, 1, 1);
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat& cell = this->at(i,j);
        
        T max;
        unsigned int maxrow, maxcol;
        
        for (int row = 0; row < cell.rows; row++)
        {
            for (int col = 0; col < cell.cols; col++)
            {
                T value = this->at(i,j,row,col); // query value
                if (value > max)
                {
                    max = value;
                    maxrow = row;
                    maxcol = col;
                }
            }
        }
        
        gargmax.at(i,j,0,0) = maxrow;
        gargmax.at(i,j,0,1) = maxcol;
    }
}


template<typename T>
void GridMat<T>::argmin(GridMat<T>& gargmin)
{
    gargmin.create(m_crows, m_ccols, 1, 1);
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat& cell = this->at(i,j);
        
        T min;
        unsigned int minrow, mincol;
        
        for (int row = 0; row < cell.rows; row++)
        {
            for (int col = 0; col < cell.cols; col++)
            {
                T value = this->at(i,j,row,col); // query value
                if (value > min)
                {
                    min = value;
                    minrow = row;
                    mincol = col;
                }
            }
        }
        
        gargmin.at(i,j,0,0) = minrow;
        gargmin.at(i,j,0,1) = mincol;
    }
}


template<typename T>
void GridMat<T>::saveFS(const std::string & filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);

	fs << "crows" << (int) m_crows;
	fs << "ccols" << (int) m_ccols;

	for (unsigned int row = 0; row < m_crows; row++)
	{
		for (int col = 0; col < m_ccols; col++)
		{
			std::stringstream ss;
			ss << "d" << row << col;
			fs << ss.str() << this->at(row,col);
		}
	}

	fs.release();
}

template<typename T>
void GridMat<T>::release()
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        this->at(i,j).release();
    }
}

template<typename T>
bool GridMat<T>::accessible(unsigned int i, unsigned int j) const
{
    return (i * m_ccols + j) <= m_grid.size() - 1;
}

template<typename T>
void GridMat<T>::show(const char* windowName)
{
    cv::namedWindow(windowName);
    cv::Mat_<T> img (m_rows, m_cols);
    
    std::cout << m_cols << " " << m_rows << std::endl;
    
    unsigned int y = 0;
    for (int i = 0; i < m_crows; i++)
    {
        unsigned int x = 0;
        int j = 0;
        for ( ; j < m_ccols; j++)
        {
            cv::Rect roi (x, y, this->at(i,j).cols, this->at(i,j).rows);
            cv::Mat_<T> roiImg (img, roi);
            this->at(i,j).copyTo(roiImg);
            
            rectangle(img, cv::Point(x,y), cv::Point(x + this->at(i,j).cols, y + this->at(i,j).rows), cv::Scalar(255,0,0));
            
            x += this->at(i,j).cols + 1;
        }
        y += this->at(i,j).rows + 1;
    }
    
    cv::imshow(windowName, img);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const GridMat<T>& gm)
{
    for (int i = 0; i < gm.crows(); i++) for (int j = 0; j < gm.ccols(); j++)
    {
        os << gm.at(i,j) << std::endl;
    }
    
    return os;
}




