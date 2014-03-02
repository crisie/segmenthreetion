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

GridMat::GridMat(unsigned int crows, unsigned int ccols, unsigned int helems, unsigned int welems, int matType)
: m_crows(crows), m_ccols(ccols)
{
    m_grid.resize( m_crows * m_ccols );
    for (unsigned int row = 0; row < m_crows; row++) for (unsigned int col = 0; col < m_ccols; col++)
    {
        this->set(cv::Mat(helems, welems, matType), row, col);
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

GridMat::GridMat(GridMat& other, cv::Mat indices)
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

void GridMat::create(unsigned int crows, unsigned int ccols)
{
    m_crows = crows;
    m_ccols = ccols;
    m_grid.resize( m_crows * m_ccols );
}

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

cv::Mat GridMat::rows()
{
    cv::Mat numrows (crows(), ccols(), cv::DataType<unsigned int>::type);
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        numrows.at<unsigned int>(i,j) = m_grid[i * ccols() + j].rows;
    }
    
    return numrows;
}

cv::Mat GridMat::cols()
{
    cv::Mat numcols (crows(), ccols(), cv::DataType<unsigned int>::type);
    for (int i = 0; i < crows(); i++) for (int j = 0; j < ccols(); j++)
    {
        numcols.at<unsigned int>(i,j) = m_grid[i * ccols() + j].cols;
    }
    
    return numcols;
}

cv::Mat GridMat::at(unsigned int i, unsigned int j) const
{
    if (accessible(i,j))
        return m_grid[i * m_ccols + j];
    else
    {
        cv::Mat m;
        return m;
    }
}

cv::Mat & GridMat::get(unsigned int i, unsigned int j)
{
    return m_grid[i * m_ccols + j];
}

void GridMat::set(cv::Mat cell, unsigned int i, unsigned int j)
{
    this->at(i,j).release();
    m_grid[i * m_ccols + j] = cell;
}

bool GridMat::isEmpty()
{
    return ccols() == 0 || crows() == 0;
}

void GridMat::hconcat(GridMat other)
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
            cv::hconcat(this->at(i,j), other.at(i,j), concatenation);
            this->at(i,j).release();
            this->set(concatenation, i,j);
        }
    }
}

void GridMat::vconcat(GridMat other)
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
            cv::vconcat(this->at(i,j), other.at(i,j), concatenation);
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
        cv::hconcat(this->at(i,j), mat, this->get(i,j));
        mat.release();
    }
}

void GridMat::vconcat(cv::Mat mat, unsigned int i, unsigned int j)
{
    if (this->at(i,j).rows == 0 && this->at(i,j).cols == 0)
        this->set(mat,i,j);
    else
    {
        cv::vconcat(this->at(i,j), mat, this->get(i,j));
        mat.release();
    }
}

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
    cv::Mat img (m_rows, m_cols, this->at(0,0).type() );
    
    std::cout << m_cols << " " << m_rows << std::endl;
    
    unsigned int y = 0;
    for (int i = 0; i < m_crows; i++)
    {
        unsigned int x = 0;
        int j = 0;
        for ( ; j < m_ccols; j++)
        {
            //std::cout << i << " " << j << " " << x << "->" << x + this->at(i,j).cols << " " << y << "->" << y + this->at(i,j).rows << std::endl;
            cv::Rect roi (x, y, this->at(i,j).cols, this->at(i,j).rows);
            cv::Mat roiImg (img, roi);
            this->at(i,j).copyTo(roiImg);
            
            rectangle(img, cv::Point(x,y), cv::Point(x + this->at(i,j).cols, y + this->at(i,j).rows), cv::Scalar(255,0,0));
            //cv::imshow(windowName, img);
            //cv::waitKey();
            x += this->at(i,j).cols + 1;
        }
        y += this->at(i,j).rows + 1;
    }
    
    cv::imshow(windowName, img);
    //cv::imwrite("a.png", img);
}

std::ostream& operator<<(std::ostream& os, const GridMat& gm)
{
    for (int i = 0; i < gm.crows(); i++) for (int j = 0; j < gm.ccols(); j++)
    {
        os << gm.at(i,j) << std::endl;
    }
    
    return os;
}




