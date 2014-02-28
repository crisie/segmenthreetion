//
//  TrimodalClusterer.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "TrimodalClusterer.h"
#include "GridMat.h"

#include <string>
#include <limits>


/*
 * Constructor
 */
TrimodalClusterer::TrimodalClusterer(int C) : m_C(C)
{ }


/*
 * Computes a subject class for each bounding box
 */
void TrimodalClusterer::trainClusters(GridMat descriptors, GridMat & labels, GridMat & centers)
{
    labels  = GridMat(descriptors.crows(), descriptors.ccols());
    centers = GridMat(descriptors.crows(), descriptors.ccols());
    
    for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {      
        cv::Mat cellLabels;
        cv::Mat cellCenters;// = Mat::zeros(m_C, descriptors.at(i,j).cols, CV_32F);
        cv::kmeans(descriptors.get(i,j), m_C, cellLabels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.001), 5, cv::KMEANS_PP_CENTERS, cellCenters);
        
        labels.set(cellLabels, i, j);
        centers.set(cellCenters, i, j);
        
//        // DEBUG
//        cout << descriptors.get(i,j).row(0) << endl;
//        cout << cellLabels << endl;
//        cout << cellCenters << endl;
        
        cellLabels.release();
        cellCenters.release();
    }
}

///*
// * Computes a subject class for each bounding box
// */
//void TrimodalClusterer::clusterize(vector<GridMat> grids, vector<GridMat> gridsMasks, Mat indices, vector<Mat> & labels, vector<Mat> & centers)
//{
//    const unsigned int hp = grids[0].crows(); // Get the 0-indexed attributes, they are all the same
//    const unsigned int wp = grids[0].ccols();
//    
//    //
//    // Describe people
//    //
//    
//    vector<Mat> trimodalCellDescs; // Want to describe each cell separately
//    trimodalCellDescs.resize(hp*wp);
//    
////    vector<Mat> trimodalDescMat;
//    for (int k = 0; k < indices.rows; k++)
//    {
//        cout << "Clusterizing grid " << k << " (" << indices.at<int>(k,0) << ") ..." << endl;
//        
//        int idx = indices.at<int>(k,0);
//        GridMat & grid = grids[idx];
//        GridMat & mask = gridsMasks[idx];
//       
//        // TODO: implement color and depth descriptors
//        
////        Mat cCellDesc = computeColorCellDescriptions(grids[i]); // DISCOMMENT after TODO
////        Mat dCellDesc = computeDepthCellsDescriptions(grids[i]); // DISCOMMENT after TODO
//        vector<Mat> tCellDesc = computeThermalCellsDescriptions(grid, mask);
//        
//        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//        {
//            Mat trimodalCellDesc;
////            vconcat(cCellDesc[i*wp+j], dCellDesc[i*wp+j], tCellDesc[i*wp+j], trimodalCellDesc);   // DISCOMMENT after TODO
//            trimodalCellDesc = tCellDesc[i*wp+j]; // REMOVE after TODO
//            
////            cCellDesc[i*wp+j].release();
////            dCellDesc[i*wp+j].release();
//            tCellDesc[i*wp+j].release();
//            
//            if (trimodalCellDescs[i*wp+j].rows == 0 && trimodalCellDescs[i*wp+j].cols == 0)
//                trimodalCellDescs[i*wp+j] = trimodalCellDesc;
//            else
//            {
//                vconcat(trimodalCellDescs[i*wp+j], trimodalCellDesc, trimodalCellDescs[i*wp+j]);
//                trimodalCellDesc.release();
//            }
//        }
//        
////        Mat trimodalDesc;
////        //vconcat(cDesc, dDesc, tDesc, trimodalDesc);   // DISCOMMENT after TODO
////        trimodalDesc = tDesc;                           // REMOVE after TODO
////        
//////        cDesc.release();                              // DISCOMMENT after TODO
//////        dDesc.release();                              // DISCOMMENT after TODO
////        tDesc.release();
////    
////        // Add rows to the trimodal descriptors matrix
////        if (i < 1) trimodalDescMat = trimodalDesc;
////        else hconcat(trimodalDescMat, trimodalDesc, trimodalDescMat);
////        
////        trimodalDesc.release();
////    }
////    
////    cout << trimodalDescMat << endl;
//    }
//    
//    //
//    // Apply clusterization
//    //
//    
//    labels.resize(indices.rows);
//    for (int i = 0; i < labels.size(); i++)
//        labels[i] = Mat (hp, wp, DataType<int>::type);
//    
//    centers.resize(hp*wp);
//    
//    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//    {
//        // TODO: build correctly from the beginning to not having to transpose later :S
////        Mat trimodalCellDescT;
////        transpose(trimodalCellDescs[i*wp+j], trimodalCellDescT);
////        trimodalCellDescs[i*wp+j].release();
//
//        cv::TermCriteria criteria;
//        criteria.maxCount = 30;
//
//        Mat cellLabels;
//        kmeans(trimodalCellDescs[i*wp+j]/*trimodalCellDescT*/, m_C, cellLabels, criteria, 5, KMEANS_PP_CENTERS, centers[i*wp+j]);
//
//        for (int k = 0; k < labels.size(); k++)
//        {
//            labels[k].at<int>(i,j) = cellLabels.at<int>(k,0);
//        }
//    }
//    
//    for (int k = 0; k < labels.size(); k++)
//    {
//        cout << labels[k] << endl;
//    }
//    
////    cout << poseLabels << endl;
//}

//Mat TrimodalClusterer::clusterize(GridMat & grid, GridMat & mask, vector<Mat> cellCenters)
//{
//    const unsigned int hp = grid.crows(); // Get the 0-indexed attributes, they are all the same
//    const unsigned int wp = grid.ccols();
//    Mat predCellsLabels (hp, wp, DataType<int>::type);
//    
//   // vector<Mat> tCellDesc = computeThermalCellsDescriptions(grid, mask);
//    
//    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//    {
//        // Extract a description
//        Mat trimodalCellDesc;
////            vconcat(cCellDesc[i*wp+j], dCellDesc[i*wp+j], tCellDesc[i*wp+j], trimodalCellDesc);   // DISCOMMENT after TODO
////        trimodalCellDesc = tCellDesc[i*wp+j]; // REMOVE after TODO
//        
//        predCellsLabels.at<int>(i,j) = closestClusterCenter(trimodalCellDesc, cellCenters[i*wp+j]);
//    }
//    
//    return predCellsLabels;
//}


void TrimodalClusterer::predictClusters(GridMat descriptors, GridMat centers, GridMat & labels)
{
    labels = GridMat(descriptors.crows(), descriptors.ccols());
    
    for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {
		cout << descriptors.at(i,j) << endl;
        const cv::Mat cellDescriptors = descriptors.at(i,j);
        cv::Mat cellLabels = cv::Mat(cellDescriptors.rows, 1, cv::DataType<int>::type);
        for (int d = 0; d < cellDescriptors.rows; d++)
        {
            cellLabels.at<int>(d,0) = closestClusterCenter(cellDescriptors.row(d), centers.at(i,j));
        }
        
        labels.set(cellLabels, i,j);
    }
}


int TrimodalClusterer::closestClusterCenter(cv::Mat point, cv::Mat centers)
{
    // Compare to corresponding cell's cluster centers  
    int closest;
    float mindist = numeric_limits<float>::max(); // mindist to infinite
    
    for (int c = 0; c < centers.rows; c++)
    {
        cv::Mat elemwiseDist;
        cv::subtract(point, centers.row(c), elemwiseDist);
        float dist = sum(elemwiseDist)[0];
        elemwiseDist.release();
        if (dist < mindist)
        {
            closest = c;
            mindist = dist;
        }
    }
    
    return closest;
}