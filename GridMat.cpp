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


GridMat::GridMat(unsigned int crows, unsigned int ccols) : m_crows(crows), m_ccols(ccols)
{    
    m_grid.resize( m_crows * m_ccols );
}


GridMat::GridMat(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems, int type)
: m_crows(crows), m_ccols(ccols)
{
    m_grid.resize( m_crows * m_ccols );
    for (unsigned int row = 0; row < m_crows; row++) for (unsigned int col = 0; col < m_ccols; col++)
    {
        this->set(cv::Mat(helems, welems, type), row, col);
    }
}


GridMat::GridMat(cv::Mat mat, unsigned int crows, unsigned int ccols)
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
        cv::Mat roi(mat,cell);
        //copyMakeBorder(roi, roi, margin, margin, margin, margin, BORDER_CONSTANT, 0);
        this->set(roi,i,j);
    }
}


GridMat::GridMat(GridMat& other, cv::Mat indices, int dim, bool logical)
{
    setIndexedCellElements(other, indices, dim, logical);
}

GridMat GridMat::getIndexedCellElements(cv::Mat indices, int dim, bool logical)
{
    if (logical)
        return getIndexedCellElementsLogically(indices, dim);
    else
        return getIndexedCellElementsPositionally(indices, dim);
}

void GridMat::setIndexedCellElements(GridMat& grid, cv::Mat indices, int dim, bool logical)
{
    if (logical)
        setIndexedCellElementsLogically(grid, indices, dim);
    else
        setIndexedCellElementsPositionally(grid, indices, dim);
}

GridMat GridMat::getIndexedCellElementsLogically(cv::Mat indices, int dim)
{
    GridMat indexed (crows(), ccols());
    int nelems = cv::sum(indices).val[0];
    
    for (int i = 0; i < crows(); i++) for (int j = 0; j < crows(); j++)
    {
        cv::Mat cellPartition (dim == 0 ? nelems : this->at(i,j).rows,
                               dim == 0 ? this->at(i,j).cols : nelems,
                               this->at(i,j).type());
        
        unsigned int n = 0;
        if (dim == 0)
        {
            for (int r = 0; r < this->at(i,j).rows; r++)
            {
                bool included = (indices.rows > 1) ? indices.at<int>(r,0) : indices.at<int>(0,r);
                if (included) this->at(i,j).row(r).copyTo(cellPartition.row(n++));
            }
        }
        else
        {
            for (int c = 0; c < this->at(i,j).cols; c++)
            {
                bool included = (indices.rows > 1) ? indices.at<int>(c,0) : indices.at<int>(0,c);
                if (included) this->at(i,j).col(c).copyTo(cellPartition.col(n++));
            }
        }
        
        indexed.set(cellPartition, i, j);
    }
}

void GridMat::setIndexedCellElementsLogically(GridMat grid, cv::Mat indices, int dim)
{
    int nelems = (indices.rows > 1) ? indices.rows : indices.cols;
    
    for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.crows(); j++)
    {
        unsigned int n = 0;
        for (int k = 0; k < nelems; k++)
        {
            bool included = (indices.rows > 1) ? indices.at<int>(k,0) : indices.at<int>(0,k);
            if (included)
            {
                if (dim == 0)
                    grid.at(i,j).row(n++).copyTo(this->at(i,j).row(k));
                else
                    grid.at(i,j).col(n++).copyTo(this->at(i,j).col(k));
            }
        }
        
    }
}

GridMat GridMat::getIndexedCellElementsPositionally(cv::Mat indices, int dim)
{
    GridMat indexed (crows(), ccols());
    
    int nelems = (indices.rows > 1) ? indices.rows : indices.cols;
    
    for (int i = 0; i < crows(); i++) for (int j = 0; j < crows(); j++)
    {
        cv::Mat cellPartition (dim == 0 ? nelems : this->at(i,j).rows,
                               dim == 0 ? this->at(i,j).cols : nelems,
                               this->at(i,j).type());
        
        for (int k = 0; k < nelems; k++)
        {
            unsigned int idx = (indices.rows > 1) ? indices.at<int>(k,0) : indices.at<int>(0,k);
            if (dim == 0) // row indexing
                this->at(i,j).row(idx).copyTo(cellPartition.row(k));
            else // column indexing
                this->at(i,j).col(idx).copyTo(cellPartition.col(k));
        }
        
        indexed.set(cellPartition, i, j);
    }
    
    return indexed;
}

void GridMat::setIndexedCellElementsPositionally(GridMat grid, cv::Mat indices, int dim)
{
    GridMat indexed (crows(), ccols());
    
    int nelems = (indices.rows > 1) ? indices.rows : indices.cols;
    
    for (int i = 0; i < crows(); i++) for (int j = 0; j < crows(); j++)
    {
        for (int k = 0; k < nelems; k++)
        {
            unsigned int idx = (indices.rows > 1) ? indices.at<int>(k,0) : indices.at<int>(0,k);
            if (dim == 0) // row indexing
                grid.at(i,j).row(k).copyTo(this->at(i,j).row(idx));
            else // column indexing
                grid.at(i,j).col(k).copyTo(this->at(i,j).col(idx));
        }
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


void GridMat::create(unsigned int crows, unsigned int ccols)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
}


//template<typename T>
//void GridMat::create(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems)
//{
//    m_crows = crows;
//    m_ccols = ccols;
//    m_grid.resize( m_crows * m_ccols );
//    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
//    {
//        this->set(cv::Mat_<T>(helems, welems), i, j);
//    }
//}


void GridMat::init(GridMat & other)
{
    this->m_crows = other.crows();
    this->m_ccols = other.ccols();
    
    this->m_grid.resize( m_crows * m_ccols );
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        this->set(other.at(i,j), i, j);
    }
}


void GridMat::copyTo(cv::Mat mat, unsigned int i, unsigned int j)
{
    mat.copyTo(m_grid[i * ccols() + j]);
}


unsigned int GridMat::crows() const
{
//    std::cout << m_crows << std::endl;
    return m_crows;
}


unsigned int GridMat::ccols() const
{
//    std::cout << m_ccols << std::endl;
    return m_ccols;
}


unsigned int GridMat::rows(unsigned int i, unsigned int j)
{
    return m_grid[i * ccols() + j].rows;
}


unsigned int GridMat::cols(unsigned int i, unsigned int j)
{
    return m_grid[i * ccols() + j].cols;
}


cv::Mat& GridMat::at(unsigned int i, unsigned int j)
{
    return m_grid[i * m_ccols + j];
}

//template<typename T>
//T& GridMat::at(unsigned int i, unsigned int j, unsigned int row, unsigned int col)
//{
//    return m_grid[i * m_ccols + j].template at<T>(row,col);
//}


cv::Mat GridMat::get(unsigned int i, unsigned int j) const
{
    return m_grid[i * m_ccols + j];
}

void GridMat::set(cv::Mat cell, unsigned int i, unsigned int j)
{
    this->at(i,j).release();
    m_grid[i * m_ccols + j] = cell;
}

GridMat GridMat::vget(cv::Mat indices, bool logical)
{
    return this->getIndexedCellElements(indices, 0, logical);
}

GridMat GridMat::hget(cv::Mat indices, bool logical)
{
    return this->getIndexedCellElements(indices, 1, logical);
}

void GridMat::vset(GridMat grid, cv::Mat indices, bool logical)
{
    this->setIndexedCellElements(grid, indices, 0, logical);
}

void GridMat::hset(GridMat grid, cv::Mat indices, bool logical)
{
    this->setIndexedCellElements(grid, indices, 1, logical);
}


bool GridMat::isEmpty()
{
    return ccols() == 0 || crows() == 0;
}


void GridMat::hconcat(GridMat& other)
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


void GridMat::vconcat(GridMat& other)
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


void GridMat::hconcat(cv::Mat mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->set(mat,i,j);
    else
    {
        cv::hconcat(this->at(i,j), mat, this->at(i,j));
        mat.release();
    }
}


void GridMat::vconcat(cv::Mat mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->set(mat,i,j);
    else
    {
        cv::vconcat(this->at(i,j), mat, this->at(i,j));
        mat.release();
    }
}



void GridMat::mean(GridMat& gmean, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat meanCol;
        cv::reduce(this->at(i,j), meanCol, dim, CV_REDUCE_AVG); // dim: 0 row-wise, 1 column-wise
        gmean.set(meanCol, i, j);
    }
}



void GridMat::max(GridMat& gmax, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MAX); // dim: 0 row-wise, 1 column-wise
        gmax.set(col, i, j);
    }
}


void GridMat::min(GridMat& gmin, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MIN); // dim: 0 row-wise, 1 column-wise
        gmin.set(col, i, j);
    }
}


void GridMat::sum(GridMat&gsum, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_SUM); // dim: 0 row-wise, 1 column-wise
        gsum.set(col, i, j);
    }
}


//template<typename T>
//void GridMat::argmax(GridMat& gargmax)
//{
//    gargmax.create<T>(m_crows, m_ccols, 1, 1);
//    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
//    {
//        cv::Mat& cell = this->at(i,j);
//        
//        T max;
//        unsigned int maxrow, maxcol;
//        
//        for (int row = 0; row < cell.rows; row++)
//        {
//            for (int col = 0; col < cell.cols; col++)
//            {
//                T value = this->at<T>(i,j,row,col); // query value
//                if (value > max)
//                {
//                    max = value;
//                    maxrow = row;
//                    maxcol = col;
//                }
//            }
//        }
//        
//        gargmax.at<T>(i,j,0,0) = maxrow;
//        gargmax.at<T>(i,j,0,1) = maxcol;
//    }
//}
//
//template<typename T>
//void GridMat::argmin(GridMat& gargmin)
//{
//    gargmin.create<T>(m_crows, m_ccols, 1, 1);
//    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
//    {
//        cv::Mat& cell = this->at(i,j);
//        
//        T min;
//        unsigned int minrow, mincol;
//        
//        for (int row = 0; row < cell.rows; row++)
//        {
//            for (int col = 0; col < cell.cols; col++)
//            {
//                T value = this->at<T>(i,j,row,col); // query value
//                if (value > min)
//                {
//                    min = value;
//                    minrow = row;
//                    mincol = col;
//                }
//            }
//        }
//        
//        gargmin.at<T>(i,j,0,0) = minrow;
//        gargmin.at<T>(i,j,0,1) = mincol;
//    }
//}


void GridMat::saveFS(const std::string & filename)
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


void GridMat::release()
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        this->at(i,j).release();
    }
}


bool GridMat::accessible(unsigned int i, unsigned int j) const
{
    return (i * m_ccols + j) <= m_grid.size() - 1;
}


void GridMat::show(const char* windowName)
{
    cv::namedWindow(windowName);
    cv::Mat img (m_rows, m_cols, this->at(0,0).type());
    
    std::cout << m_cols << " " << m_rows << std::endl;
    
    unsigned int y = 0;
    for (int i = 0; i < m_crows; i++)
    {
        unsigned int x = 0;
        int j = 0;
        for ( ; j < m_ccols; j++)
        {
            cv::Rect roi (x, y, this->at(i,j).cols, this->at(i,j).rows);
            cv::Mat roiImg (img, roi);
            this->at(i,j).copyTo(roiImg);
            
            rectangle(img, cv::Point(x,y), cv::Point(x + this->at(i,j).cols, y + this->at(i,j).rows), cv::Scalar(255,0,0));
            
            x += this->at(i,j).cols + 1;
        }
        y += this->at(i,j).rows + 1;
    }
    
    cv::imshow(windowName, img);
}


std::ostream& operator<<(std::ostream& os, const GridMat& gm)
{
    for (int i = 0; i < gm.crows(); i++) for (int j = 0; j < gm.ccols(); j++)
    {
        os << gm.get(i,j) << std::endl;
    }
    
    return os;
}




