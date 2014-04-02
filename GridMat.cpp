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


// Instantiation of template member functions
// -----------------------------------------------------------------------------
template cv::Mat GridMat::findNonZero<unsigned char>();

template void GridMat::create<unsigned char>(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems);
template void GridMat::create<int>(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems);
template void GridMat::create<float>(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems);
template void GridMat::create<double>(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems);

template unsigned char& GridMat::at<unsigned char>(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
template int& GridMat::at<int>(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
template float& GridMat::at<float>(unsigned int i, unsigned int j, unsigned int row, unsigned int col);
template double& GridMat::at<double>(unsigned int i, unsigned int j, unsigned int row, unsigned int col);

template void GridMat::argmax<unsigned char>(GridMat& gargmax);
template void GridMat::argmax<int>(GridMat& gargmax);
template void GridMat::argmax<float>(GridMat& gargmax);
template void GridMat::argmax<double>(GridMat& gargmax);

template void GridMat::argmin<unsigned char>(GridMat& gargmin);
template void GridMat::argmin<int>(GridMat& gargmin);
template void GridMat::argmin<float>(GridMat& gargmin);
template void GridMat::argmin<double>(GridMat& gargmin);

template void GridMat::setTo<unsigned char>(unsigned char value, unsigned int i, unsigned j);
template void GridMat::setTo<int>(int value, unsigned int i, unsigned int j);
template void GridMat::setTo<float>(float value, unsigned int i, unsigned int j);
template void GridMat::setTo<double>(double value, unsigned int i, unsigned int j);

template void GridMat::setTo<unsigned char>(unsigned char value, unsigned int i, unsigned int j, cv::Mat mask);
template void GridMat::setTo<int>(int value, unsigned int i, unsigned int j, cv::Mat mask);
template void GridMat::setTo<float>(float value, unsigned int i, unsigned int j, cv::Mat mask);
template void GridMat::setTo<double>(double value, unsigned int i, unsigned int j, cv::Mat mask);

template void GridMat::convertToMat<unsigned char>(cv::Mat& mat);
template void GridMat::convertToMat<int>(cv::Mat& mat);
template void GridMat::convertToMat<float>(cv::Mat& mat);
template void GridMat::convertToMat<double>(cv::Mat& mat);

template cv::Mat GridMat::convertToMat<unsigned char>();
template cv::Mat GridMat::convertToMat<int>();
template cv::Mat GridMat::convertToMat<float>();
template cv::Mat GridMat::convertToMat<double>();
// -----------------------------------------------------------------------------


GridMat::GridMat(unsigned int crows, unsigned int ccols) : m_crows(crows), m_ccols(ccols)
{    
    m_grid.resize(m_crows * m_ccols);
}


GridMat::GridMat(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems, int type)
: m_crows(crows), m_ccols(ccols)
{
    m_grid.resize(m_crows * m_ccols);
    for (unsigned int row = 0; row < m_crows; row++) for (unsigned int col = 0; col < m_ccols; col++)
    {
        cv::Mat mat (helems, welems, type);
        this->assign(mat, row, col);
    }
}


GridMat::GridMat(cv::Mat mat, unsigned int crows, unsigned int ccols)
: m_crows(crows), m_ccols(ccols)
{
    m_rows = mat.rows;
    m_cols = mat.cols;
    
    m_grid.resize(m_crows * m_ccols);
    
    int a = floorf(((float) m_rows) / m_crows);     // + (m_rows % m_crows)/m_crows;
    int b = floorf(((float) m_cols) / m_ccols);     // + (m_cols % m_ccols)/m_ccols;
    //unsigned int cellh, cellw;
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        //cellh = ((j+1) * b - 1) - (j * b);
        //cellw = ((i+1) * a - 1) - (i * a);
        cv::Rect cell = cv::Rect((j * b), (i * a), b, a);// cellh, cellw);
        cv::Mat roi(mat,cell);
        //copyMakeBorder(roi, roi, margin, margin, margin, margin, BORDER_CONSTANT, 0);
        //cv::namedWindow("grid");
        //cv::imshow("grid", roi);
        //cv::waitKey();
        this->assign(roi,i,j);
    }
}

GridMat::GridMat(const GridMat& other)
{
    m_crows = other.m_crows;
    m_ccols = other.m_ccols;
    m_grid = other.m_grid;
}

GridMat::GridMat(GridMat& other, cv::Mat indices, int dim, bool logical)
{
    m_crows = other.m_crows;
    m_ccols = other.m_ccols;
    
    m_grid.resize(other.crows() * other.ccols());
    
    setIndexedCellElements(other, indices, dim, logical);
}

GridMat::GridMat(GridMat& other, GridMat indices, int k, bool inverse)
{
    m_crows = other.m_crows;
    m_ccols = other.m_ccols;
    
    m_grid.resize( m_crows * m_ccols );
    
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        if (!inverse)
        {
            indexMatElements(other.at(i,j), at(i,j), indices.at(i,j) == k);
        }
        else // inverse indexing: all but k-th index
        {
            indexMatElements(other.at(i,j), at(i,j), indices.at(i,j) != k);
        }
    }
}

GridMat::GridMat(GridMat& other, GridMat indices, bool logical)
{
    m_crows = other.m_crows;
    m_ccols = other.m_ccols;
    
    m_grid.resize( m_crows * m_ccols );
    
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        indexMatElements(other.at(i,j), at(i,j), indices.at(i,j), logical);
    }
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

GridMat GridMat::getIndexedCellElementsLogically(cv::Mat logicals, int dim)
{
    GridMat indexed (crows(), ccols());
    int nelems = cv::sum(logicals).val[0];
    
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
                unsigned char included = (logicals.rows > 1) ? logicals.at<int>(r,0) : logicals.at<int>(0,r);
                if (included) this->at(i,j).row(r).copyTo(cellPartition.row(n++));
            }
        }
        else
        {
            for (int c = 0; c < this->at(i,j).cols; c++)
            {
                unsigned char included = (logicals.rows > 1) ? logicals.at<int>(c,0) : logicals.at<int>(0,c);
                if (included) this->at(i,j).col(c).copyTo(cellPartition.col(n++));
            }
        }
        
        indexed.assign(cellPartition, i, j);
    }

	return indexed;
}

void GridMat::setIndexedCellElementsLogically(GridMat& grid, cv::Mat logicals, int dim)
{
    int nelems = (logicals.rows > 1) ? logicals.rows : logicals.cols;
    for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.crows(); j++)
    {
        for (int k = 0; k < nelems; k++)
        {
            unsigned char included = (logicals.rows > 1) ? logicals.at<int>(k,0) : logicals.at<int>(0,k);
            if (included)
            {
                if (dim == 0)
                {
                    cv::Mat row = grid.at(i,j).row(k);
                    this->vconcat(row, i, j);
                }
                else
                {
                    cv::Mat col = grid.at(i,j).col(k);
                    this->vconcat(col, i, j);
                }
//                if (dim == 0)
//                    grid.at(i,j).row(n++).copyTo(this->at(i,j).row(k));
//                else
//                    grid.at(i,j).col(n++).copyTo(this->at(i,j).col(k));
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
        
        indexed.assign(cellPartition, i, j);
    }
    
    return indexed;
}

void GridMat::setIndexedCellElementsPositionally(GridMat& grid, cv::Mat indices, int dim)
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
//        this->assign(other.at(i,j), i, j);
//    }
//}


void GridMat::create(unsigned int crows, unsigned int ccols)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
}


template<typename T>
void GridMat::create(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        cv::Mat_<T> mat (helems, welems);
        this->assign(mat, i, j);
    }
}


void GridMat::init(GridMat & other)
{
    this->m_crows = other.crows();
    this->m_ccols = other.ccols();
    
    this->m_grid.resize( m_crows * m_ccols );
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        this->assign(other.at(i,j), i, j);
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

template<typename T>
T& GridMat::at(unsigned int i, unsigned int j, unsigned int row, unsigned int col)
{
    return m_grid[i * m_ccols + j].template at<T>(row,col);
}


cv::Mat GridMat::get(unsigned int i, unsigned int j) const
{
    return m_grid[i * m_ccols + j];
}

void GridMat::assign(cv::Mat cell, unsigned int i, unsigned int j)
{
    m_grid[i * m_ccols + j] = cell;
}

void GridMat::set(GridMat& other)
{
    this->release();
    
    m_crows = other.m_crows;
    m_cols = other.m_ccols;
    
    for (int i = 0; i < m_crows; i++) for (int j = 0; j < m_ccols; j++)
    {
        m_grid[i * m_ccols + j] = other.at(i,j);
    }
}

void GridMat::set(GridMat src, GridMat indices, int k)
{
    if (isEmpty())
    {
        m_crows = src.crows();
        m_ccols = src.ccols();
        m_grid.resize(m_crows * m_cols);
    }
        
    for (int i = 0; i < indices.crows(); i++) for (int j = 0; j < indices.ccols(); j++)
    {
        this->setMatElements(src.at(i,j), this->at(i,j), indices.at(i,j) == k);
    }
}

void GridMat::copyTo(GridMat& dst, GridMat indices, int k)
{
    if (dst.isEmpty())
    {
        dst.create(m_crows, m_ccols);
    }
    
    for (int i = 0; i < indices.crows(); i++) for (int j = 0; j < indices.ccols(); j++)
    {
        this->copyMatElements(this->at(i,j), dst.at(i,j), indices.at(i,j) == k);
    }
}

template<typename T>
void GridMat::setTo(T value, unsigned int i, unsigned int j)
{
    if (this->at(i,j).type() != cv::DataType<T>::type)
        return;
        
    cv::Mat tmp (this->at(i,j).rows, this->at(i,j).cols, cv::DataType<T>::type);
    tmp.setTo(value);
    
    this->at(i,j).release();
    tmp.copyTo(this->at(i,j));
}

template<typename T>
void GridMat::setTo(T value, unsigned int i, unsigned int j, cv::Mat mask)
{
    if (this->at(i,j).type() != cv::DataType<T>::type)
        return;
    
    cv::Mat tmp (this->at(i,j).rows, this->at(i,j).cols, cv::DataType<T>::type);
    tmp.setTo(value);
    
    this->at(i,j).release();
    tmp.copyTo(this->at(i,j), mask);
}

void GridMat::setTo(cv::Mat mat)
{
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        this->at(i,j).assignTo(mat);
    }
}

template<typename T>
void GridMat::convertToMat(cv::Mat& mat)
{
    mat.release();
    
    int a = floorf(((float) m_rows) / m_crows);     // + (m_rows % m_crows)/m_crows;
    int b = floorf(((float) m_cols) / m_ccols);     // + (m_cols % m_ccols)/m_ccols;
    
    mat.create(a * crows(), b * crows(), cv::DataType<T>::type);
    
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        cv::Mat cell;
        this->at(i,j).convertTo(cell, cv::DataType<T>::type);
        
        cv::Mat roi (mat, cv::Rect((j * b), (i * a), b, a));
        cell.copyTo(roi);
    }
}

template<typename T>
cv::Mat GridMat::convertToMat()
{
    cv::Mat tmp;
    this->convertToMat<T>(tmp);
    
    return tmp;
}

GridMat GridMat::convertToSparse(GridMat indices)
{
    GridMat g;
    convertToSparse(indices, g);
    
    return g;
}

void GridMat::convertToSparse(GridMat indices, GridMat& sparseGridMat)
{
    assert (crows() != indices.crows() || ccols() != indices.ccols());
    
    cv::Mat counts(crows(), ccols(), cv::DataType<int>::type);
    counts.setTo(0);
    
    sparseGridMat.release();
    sparseGridMat.create(indices.crows(), indices.ccols());
    
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        if (indices.at(i,j).rows > 1)
            sparseGridMat.at(i,j).create(indices.at(i,j).rows, this->at(i,j).cols, this->at(i,j).type());
        else
            sparseGridMat.at(i,j).create(this->at(i,j).rows, indices.at(i,j).cols, this->at(i,j).type());
        
        for (int k = 0; k < indices.at(i,j).rows; k++)
        {
            if (indices.at<unsigned char>(i,j,k,0))
            {
                if (indices.at(i,j).rows > 1)
                    this->at(i,j).row(counts.at<int>(i,j)++).copyTo(sparseGridMat.at(i,j).row(k));
                else
                    this->at(i,j).col(counts.at<int>(i,j)++).copyTo(sparseGridMat.at(i,j).col(k));
            }
            else
            {
                cv::Mat mat;
                if (indices.at(i,j).rows > 1)
                {
                    mat.create(1, this->at(i,j).cols, this->at(i,j).type());
                    mat.setTo(0);
                    mat.copyTo(sparseGridMat.at(i,j).row(k));
                }
                else
                {
                    mat.create(this->at(i,j).rows, 1, this->at(i,j).type());
                    mat.setTo(0);
                    mat.copyTo(sparseGridMat.at(i,j).col(k));
                }
            }
        }
    }
}

GridMat GridMat::convertToDense(GridMat indices)
{
    GridMat g;
    convertToDense(indices, g);
    
    return g;
}

void GridMat::convertToDense(GridMat indices, GridMat& denseGridMat)
{
    assert (crows() != indices.crows() || ccols() != indices.ccols());
    
    denseGridMat = GridMat(*this, indices, 0, true);
}


void GridMat::hserial(cv::Mat& serial)
{
    serial = this->at(0,0).clone();
    for (int i = 0; i < crows(); i++) for (int j = 1; j < ccols(); j++)
    {
        cv::hconcat(serial, this->at(i,j), serial);
    }
}

void GridMat::vserial(cv::Mat& serial)
{
    serial = this->at(0,0).clone();
    for (int i = 0; i < crows(); i++) for (int j = 1; j < ccols(); j++)
    {
        cv::vconcat(serial, this->at(i,j), serial);
    }
}

void GridMat::normalize(GridMat& normalized)
{
    normalized.create(crows(), ccols());
    
    double minValDbl, maxValDbl;
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        cv::minMaxIdx(this->at(i,j), &minValDbl, &maxValDbl);
        float minVal = static_cast<float>(minValDbl);
        float maxVal = static_cast<float>(maxValDbl);
        
        cv::Mat tmp;
        this->at(i,j).convertTo(tmp, cv::DataType<float>::type);
        
        cv::Mat normalizedCell = ((this->at(i,j) - minVal) / (maxVal - minVal));
        normalizedCell.copyTo(normalized.at(i,j));
    }
}

GridMat GridMat::normalize()
{
    GridMat g;
    normalize(g);
    return g;
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

cv::Rect GridMat::getCellCoordinates(unsigned int i, unsigned int j)
{
    int a = floorf(((float) m_rows) / m_crows);     // + (m_rows % m_crows)/m_crows;
    int b = floorf(((float) m_cols) / m_ccols);     // + (m_cols % m_ccols)/m_ccols;
    
    return cv::Rect((j * b), (i * a), b, a);
}

void GridMat::hconcat(GridMat& other)
{
    if (this->isEmpty())
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        
        m_grid = other.m_grid;
    }
    else
    {
        // Concatenate
        for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
        {
            if (this->at(i,j).empty())
            {
                this->assign(other.at(i,j), i, j);
            }
            else
            {
                cv::hconcat(this->at(i,j), other.at(i,j), this->at(i,j));
            }
        }
    }
}


void GridMat::vconcat(GridMat& other)
{
    if (this->isEmpty())
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        
        m_grid = other.m_grid;
    }
    else
    {
        // Concatenate
        for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
        {
            if (this->at(i,j).empty())
            {
                this->assign(other.at(i,j), i, j);
            }
            else
            {
                cv::vconcat(this->at(i,j), other.at(i,j), this->at(i,j));
            }
        }
    }
}


void GridMat::hconcat(cv::Mat& mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->assign(mat,i,j);
    else
    {
        cv::hconcat(this->at(i,j), mat, this->at(i,j));
    }
}


void GridMat::vconcat(cv::Mat& mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->assign(mat,i,j);
    else
    {
        cv::vconcat(this->at(i,j), mat, this->at(i,j));
    }
}



void GridMat::mean(GridMat& gmean, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat meanCol;
        cv::reduce(this->at(i,j), meanCol, dim, CV_REDUCE_AVG); // dim: 0 row-wise, 1 column-wise
        gmean.assign(meanCol, i, j);
    }
}



void GridMat::max(GridMat& gmax, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MAX); // dim: 0 row-wise, 1 column-wise
        gmax.assign(col, i, j);
    }
}


void GridMat::min(GridMat& gmin, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat col;
        cv::reduce(this->at(i,j), col, dim, CV_REDUCE_MIN); // dim: 0 row-wise, 1 column-wise
        gmin.assign(col, i, j);
    }
}


void GridMat::sum(GridMat& gsum, int dim)
{
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat mat;
        this->at(i,j).convertTo(mat, CV_32F);
        
        cv::Mat col;
        cv::reduce(mat, col, dim, CV_REDUCE_SUM); // dim: 0 row-wise, 1 column-wise
        gsum.assign(col, i, j);
    }
}

template<typename T>
cv::Mat GridMat::findNonZero()
{
    cv::Mat nonZerosMat(m_crows, m_ccols, cv::DataType<unsigned char>::type);
    
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        bool nonZeroFound = false;
        for (unsigned int row = 0; row < this->at(i,j).rows && !nonZeroFound; row++)
            for (unsigned int col = 0; col < this->at(i,j).cols && !nonZeroFound; col++)
                nonZeroFound = this->at(i,j).at<T>(row,col) > 0;
        
        nonZeroFound ? nonZerosMat.at<unsigned char>(i,j) = 1 : nonZerosMat.at<unsigned char>(i,j) = 0;
    }
    
    return nonZerosMat;
}


template<typename T>
void GridMat::argmax(GridMat& gargmax)
{
    gargmax.release();
    gargmax.create<int>(m_crows, m_ccols, 1, 2);
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat& cell = this->at(i,j);
        
        T max;
        int maxrow, maxcol;
        
        for (int row = 0; row < cell.rows; row++)
        {
            for (int col = 0; col < cell.cols; col++)
            {
                T value = this->at<T>(i,j,row,col); // query value
                if (value > max)
                {
                    max = value;
                    maxrow = row;
                    maxcol = col;
                }
            }
        }
        
        gargmax.at<int>(i,j,0,0) = maxrow;
        gargmax.at<int>(i,j,0,1) = maxcol;
    }
}

template<typename T>
void GridMat::argmin(GridMat& gargmin)
{
    gargmin.release();
    gargmin.create<int>(m_crows, m_ccols, 1, 2);
    for (unsigned int i = 0; i < m_crows; i++) for (unsigned int j = 0; j < m_ccols; j++)
    {
        cv::Mat& cell = this->at(i,j);
        
        T min;
        int minrow, mincol;
        
        for (int row = 0; row < cell.rows; row++)
        {
            for (int col = 0; col < cell.cols; col++)
            {
                T value = this->at<T>(i,j,row,col); // query value
                if (value > min)
                {
                    min = value;
                    minrow = row;
                    mincol = col;
                }
            }
        }
        
        gargmin.at<int>(i,j,0,0) = minrow;
        gargmin.at<int>(i,j,0,1) = mincol;
    }
}


void GridMat::save(const std::string & filename)
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


void GridMat::load(const std::string & filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << filename << " not found" << endl;
        return;
    }
   
	(int) fs["crows"] >> m_crows;
	(int) fs["ccols"] >> m_ccols;
    
	for (unsigned int row = 0; row < m_crows; row++)
	{
		for (int col = 0; col < m_ccols; col++)
		{
			std::stringstream ss;
			ss << "d" << row << col;
            cv::Mat aux;
            fs[ss.str().c_str()] >> aux;
			this->assign(aux, row, col);
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

void GridMat::setMatElements(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatElementsLogically(src, dst, indices);
    else
        setMatElementsPositionally(src, dst, indices);
}

void GridMat::setMatElementsLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
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

void GridMat::setMatElementsPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    bool rowwise = indexes.rows > 1;
    
    int n = rowwise ? indexes.rows : indexes.cols;
    for (int i = 0; i < n; i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(i).copyTo(dst.row(idx)) : src.col(i).copyTo(dst.col(idx));
    }
}

void GridMat::copyMatElements(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        setMatElementsLogically(src, dst, indices);
    else
        setMatElementsPositionally(src, dst, indices);
}

void GridMat::copyMatElementsLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = logicals.rows > 1;
    
    for (int i = 0; i < rowwise ? logicals.rows : logicals.cols; i++)
    {
        unsigned char indexed = rowwise ?
        logicals.at<unsigned char>(i,0) : logicals.at<unsigned char>(0,i);
        if (indexed)
            rowwise ? src.row(i).copyTo(dst.row(i)) : src.col(i).copyTo(dst.col(i));
    }
}

void GridMat::copyMatElementsPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indexes)
{
    if (dst.empty())
        dst.create(src.rows, src.cols, src.type());
    
    bool rowwise = indexes.rows > 1;
    
    for (int i = 0; i < rowwise ? indexes.rows : indexes.cols; i++)
    {
        int idx = rowwise ? indexes.at<int>(i,0) : indexes.at<int>(i,0);
        rowwise ? src.row(idx).copyTo(dst.row(idx)) : src.col(idx).copyTo(dst.col(idx));
    }
}

void GridMat::indexMatElements(cv::Mat src, cv::Mat& dst, cv::Mat indices, bool logical)
{
    if (logical)
        indexMatElementsLogically(src, dst, indices);
    else
        indexMatElementsPositionally(src, dst, indices);
}

void GridMat::indexMatElementsLogically(cv::Mat src, cv::Mat& dst, cv::Mat logicals)
{
    if (logicals.rows > 1) // row-wise
        dst.create(0, src.cols, src.type());
    else // col-wise
        dst.create(src.rows, 0, src.type());
    
    for (int i = 0; i < (logicals.rows > 1 ? logicals.rows : logicals.cols); i++)
    {
        if (logicals.rows > 1)
        {
            if (logicals.at<unsigned char>(i,0))
            {
                dst.push_back(src.row(i));
            }
        }
        else
        {
            if (logicals.at<unsigned char>(0,i))
                dst.push_back(src.col(i));
        }
    }
}

void GridMat::indexMatElementsPositionally(cv::Mat src, cv::Mat& dst, cv::Mat indices)
{
    dst.release();
    
    if (indices.rows > 1) // row-wise
        dst.create(indices.rows, src.cols, src.type());
    else // col-wise
        dst.create(src.rows, indices.cols, src.type());
    
    for (int i = 0; i < (indices.rows > 1 ? indices.rows : indices.cols); i++)
    {
        int idx = indices.rows > 1 ? indices.at<int>(i,0) : indices.at<int>(0,i);
        
        if (indices.rows > 1)
            src.row(idx).copyTo(dst.row(i));
        else
            src.col(idx).copyTo(dst.col(i));
    }
}



