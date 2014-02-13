//
//  TrimodalFeatureExtractor.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "TrimodalFeatureExtractor.h"

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread.hpp>
#include <boost/timer.hpp>

TrimodalFeatureExtractor::TrimodalFeatureExtractor(int hp, int wp) : m_hp(hp), m_wp(wp)
{ }

void TrimodalFeatureExtractor::setThermalData(vector<GridMat> grids, vector<GridMat> masks)
{
    m_ThermalGrids = grids;
    m_ThermalMasks = masks;
}

void TrimodalFeatureExtractor::setDepthData(vector<GridMat> grids, vector<GridMat> masks)
{
    m_DepthGrids = grids;
    m_DepthMasks = masks;
}

void TrimodalFeatureExtractor::setThermalParam(ThermalParametrization thermalParam)
{
    m_ThermalParam = thermalParam;
}

void TrimodalFeatureExtractor::setDepthParam(DepthParametrization depthParam)
{
    m_DepthParam = depthParam;
}

void TrimodalFeatureExtractor::describeThermalIntesities(const cv::Mat cell, const cv::Mat cellMask, cv::Mat & tIntensitiesHist)
{
    int ibins = m_ThermalParam.ibins;
    
    // Create an histogram for the cell region of blurred intensity values
    int histSize[] = { (int) ibins };
    int channels[] = { 0 }; // 1 channel, number 0
    float tranges[] = { 0, 256 }; // thermal intensity values range: [0, 256)
    const float* ranges[] = { tranges };
    
    cv::Mat tmpHist;
    cv::calcHist(&cell, 1, channels, cellMask, tmpHist, 1, histSize, ranges, true, false);
    cv::transpose(tmpHist, tmpHist);
    
    hypercubeNorm(tmpHist, tIntensitiesHist);
    tmpHist.release();
}

void TrimodalFeatureExtractor::describeThermalGradOrients(const cv::Mat cell, const cv::Mat cellMask, cv::Mat & tGradOrientsHist)
{
    cv::Mat cellSeg = cv::Mat::zeros(cell.rows, cell.cols, cell.depth());
    cell.copyTo(cellSeg, cellMask);
    
    // First derivatives
    cv::Mat cellDervX, cellDervY;
    cv::Sobel(cellSeg, cellDervX, CV_32F, 1, 0);
    cv::Sobel(cellSeg, cellDervY, CV_32F, 0, 1);
    
    cv::Mat cellGradOrients;
    cv::phase(cellDervX, cellDervY, cellGradOrients, true);
    
    int oribins = m_ThermalParam.oribins;
    
    cv::Mat tmpHist = cv::Mat::zeros(1, oribins, cv::DataType<float>::type);
    
    for (int i = 0; i < cellSeg.rows; i++) for (int j = 0; j < cellSeg.cols; j++)
    {
        float g_x = cellDervX.at<unsigned short>(i,j);
        float g_y = cellDervY.at<unsigned short>(i,j);
        
        float orientation = cellGradOrients.at<float>(i,j);
        float bin = static_cast<int>((orientation/360.0) * oribins) % oribins;
        tmpHist.at<float>(0, bin) += sqrtf(g_x * g_x + g_y * g_y);
    }
    
    cellSeg.release();
    cellDervX.release();
    cellDervY.release();
    cellGradOrients.release();
    
    hypercubeNorm(tmpHist, tGradOrientsHist);
    tmpHist.release();
}

//void TrimodalFeatureExtractor::projectiveToRealWorld(float px, float py, float pz, float & rwx, float & rwy, float & rwz)
//{
//    
//}

void TrimodalFeatureExtractor::describeNormalsOrients(const cv::Mat grid, const cv::Mat mask, cv::Mat & tNormalsOrientsHist)
{
    pcl::visualization::PCLVisualizer viz("viz");
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud ( new pcl::PointCloud<pcl::PointXYZ>() );
    pCloud->height = grid.rows;
    pCloud->width = grid.cols;
    pCloud->resize(pCloud->height * pCloud->width);

    // Easy to handle the conversion this way
    cv::Mat temp = cv::Mat(grid.size(), CV_32F);
    grid.convertTo(temp, CV_32F);
    
    float invfocal = 3.501e-3f;

    for (unsigned int y = 0; y < pCloud->height; y++) for (unsigned int x = 0; x < pCloud->width; x++)
    {
        float rwx, rwy, rwz;
        rwx = (x - 160.0) * invfocal * temp.at<float>(y,x);
        rwy = (y - 120.0) * invfocal * temp.at<float>(y,x);
        rwz = temp.at<float>(y,x);
        
        pcl::PointXYZ p(rwx, rwy, rwz);
        pCloud->at(y,x) = p;
    }
    
    viz.addPointCloud<pcl::PointXYZ>(pCloud, "sample cloud");
    viz.initCameraParameters();
    
    while (!viz.wasStopped ())
    {
        viz.spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    
    temp.release();
    
    
    
}

void TrimodalFeatureExtractor::describeThermal(GridMat & descriptors)
{
//    namedWindow("god");
    
    for (int k = 0; k < m_ThermalGrids.size(); k++)
    {
        GridMat & grid = m_ThermalGrids[k];
        GridMat & mask = m_ThermalMasks[k];
        
        for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.ccols(); j++)
        {
            cv::Mat & cell = grid.get(i,j);
            cv::Mat & cellMask = mask.get(i,j);
            
//            imshow("god", cell);
//            waitKey(0);
            
            // Intensities descriptor
            cv::Mat tIntensitiesHist;
            describeThermalIntesities(cell, cellMask, tIntensitiesHist);
        
            // Gradient orientation descriptor
            cv::Mat tGradOrientsHist;
            describeThermalGradOrients(cell, cellMask, tGradOrientsHist);
        
            // Join both descriptors in a row
            cv::Mat tHist;
            hconcat(tIntensitiesHist, tGradOrientsHist, tHist);
                    
//            cout << tHist << endl; // DEBUG
            
            // Consider the descriptor only if does not contain nans
//            if (checkRange(tHist))
//            {
                descriptors.vconcat(tHist, i, j); // row in a matrix of descriptors
//            }
        }
    }
}

void TrimodalFeatureExtractor::describeDepth(GridMat & descriptors)
{
    //    namedWindow("god");
    
    for (int k = 0; k < m_ThermalGrids.size(); k++)
    {
        GridMat & grid = m_ThermalGrids[k];
        GridMat & mask = m_ThermalMasks[k];
        
        for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.ccols(); j++)
        {
            cv::Mat & cell = grid.get(i,j);
            cv::Mat & cellMask = mask.get(i,j);
            
            //            imshow("god", cell);
            //            waitKey(0);
            
            // Normals orientation descriptor
            cv::Mat tNormalsOrientsHist;
            describeNormalsOrients(cell, cellMask, tNormalsOrientsHist);
            
            //            cout << tHist << endl; // DEBUG
            
            // Consider the descriptor only if does not contain nans
//            if (checkRange(tNormalsOrientsHist))
//            {
                descriptors.vconcat(tNormalsOrientsHist, i, j); // row in a matrix of descriptors
//            }
        }
    }
    
    for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {
        cv::Mat x = descriptors.at(i,j);
        cv::Mat mean;
        cv::Mat stddev;
        cv::meanStdDev(x, mean, stddev);
        
        cout << mean << endl;
    }
}

void TrimodalFeatureExtractor::describe(GridMat & descriptionsThermal, GridMat & descriptionsDepth)
{
    descriptionsThermal.release();
    descriptionsDepth.release();
    
    descriptionsDepth   = GridMat(m_hp, m_wp);
    descriptionsThermal = GridMat(m_hp, m_wp);
    
    describeDepth(descriptionsDepth);
    describeThermal(descriptionsThermal);
}

/*
 * Hypercube normalization
 */
void TrimodalFeatureExtractor::hypercubeNorm(cv::Mat & src, cv::Mat & dst)
{
    src.copyTo(dst);
    double z = sum(src).val[0]; // partition function :D
    dst = dst / z;
}