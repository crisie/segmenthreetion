//
//  ThermalFeatureExtractor.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "FeatureExtractor.h"
#include "ThermalFeatureExtractor.h"

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread.hpp>
#include <boost/timer.hpp>


ThermalFeatureExtractor::ThermalFeatureExtractor()
    : FeatureExtractor()
{ }


ThermalFeatureExtractor::ThermalFeatureExtractor(ThermalParametrization tParam)
    : FeatureExtractor(), m_ThermalParam(tParam)
{ }


void ThermalFeatureExtractor::setParam(ThermalParametrization thermalParam)
{
    m_ThermalParam = thermalParam;
}


void ThermalFeatureExtractor::describe(ModalityGridData& data)
{
    for (int k = 0; k < data.getGridsFrames().size(); k++)
    {
        if (k % 100 == 0) cout << 100.0 * k / data.getGridsFrames().size() << "%" <<  endl; // debug
        
        GridMat grid = data.getGridFrame(k);
        GridMat gmask = data.getGridMask(k);
        cv::Mat gvalidness = data.getValidnesses(k);

        for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.ccols(); j++)
        {
            cv::Mat tHist (1, m_ThermalParam.ibins + m_ThermalParam.oribins, CV_32F);
            tHist.setTo(std::numeric_limits<float>::quiet_NaN());
            
            if (gvalidness.at<unsigned char>(i,j))
            {
                cv::Mat& cell = grid.at(i,j);
                cv::Mat& cellMask = gmask.at(i,j);
                
                // Intensities descriptor
                cv::Mat tIntensitiesHist;
                describeThermalIntesities(cell, cellMask, tIntensitiesHist);
                
                // Gradient orientation descriptor
                cv::Mat tGradOrientsHist, dX, dY, p;
                describeThermalGradOrients(cell, cellMask, tGradOrientsHist);
                
                // Join both descriptors in a row
                hconcat(tIntensitiesHist, tGradOrientsHist, tHist);
            }
            
            data.addDescriptor(tHist, i, j); // row in a matrix of descriptors
        }
    }
}


void ThermalFeatureExtractor::describeThermalIntesities(cv::Mat cell, cv::Mat cellMask, cv::Mat & tIntensitiesHist)
{
    int ibins = m_ThermalParam.ibins;
    
    // Create an histogram for the cell region of blurred intensity values
    int histSize[] = { (int) ibins };
    int channels[] = { 0 }; // 1 channel, number 0
    float tranges[] = { 0, 256 }; // thermal intensity values range: [0, 256)
    const float* ranges[] = { tranges };
    
    cv::Mat tmpHist;
    calcHist(&cell, 1, channels, cellMask, tmpHist, 1, histSize, ranges, true, false);
    transpose(tmpHist, tmpHist);
    
    hypercubeNorm(tmpHist, tIntensitiesHist);
    tmpHist.release();
}


void ThermalFeatureExtractor::describeThermalGradOrients(cv::Mat cell, cv::Mat cellMask, cv::Mat & tGradOrientsHist)
{
    //cv::Mat cellSeg = cv::Mat::zeros(cell.rows, cell.cols, cell.type());
    //cell.copyTo(cellSeg, cellMask);
    
    // First derivatives
    cv::Mat cellDervX, cellDervY;
    cv::Mat aux = cell.clone();
    cv::Sobel(aux, cellDervX, CV_32F, 1, 0);
    cv::Sobel(aux, cellDervY, CV_32F, 0, 1);
    
    cv::Mat cellGradOrients;
    cv::phase(cellDervX, cellDervY, cellGradOrients, true);
    
    int oribins = m_ThermalParam.oribins;
    
    cv::Mat tmpHist = cv::Mat::zeros(1, oribins, cv::DataType<float>::type);
    
    for (int i = 0; i < cell.rows; i++) for (int j = 0; j < cell.cols; j++)
    {
        if (cellMask.at<unsigned char>(i,j))
        {
            float g_x = cellDervX.at<float>(i,j);//unsigned short>(i,j);
            float g_y = cellDervY.at<float>(i,j);
            
            float orientation = cellGradOrients.at<float>(i,j);
            int bin = ((int) floorf((orientation/360.0) * oribins));
            tmpHist.at<float>(0, bin) += sqrtf(g_x * g_x + g_y * g_y);
        }
    }
    
    hypercubeNorm(tmpHist, tGradOrientsHist);
}