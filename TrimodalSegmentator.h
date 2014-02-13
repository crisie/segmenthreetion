//
//  TrimodalSegmentator.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__TrimodalSegmentator__
#define __Segmenthreetion__TrimodalSegmentator__

#include "ColorParametrization.h"
#include "MotionParametrization.h"
#include "DepthParametrization.h"
#include "ThermalParametrization.h"

#include "GridMat.h"
#include "TrimodalClusterer.h"

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

class TrimodalSegmentator
{
public:
    /*
     * Constructors
     */
    TrimodalSegmentator(const unsigned int hp, const unsigned int wp, const int numClusters, const int numMixtures, const ColorParametrization cParam, const MotionParametrization mParam, const DepthParametrization dParam, const ThermalParametrization tParam);
    
    /*
     * Public methods
     */
    
    // Set the path containing the sequence in which the segmentation will be performed
    void setDataPath(string dataPath);
    
    // Run the program
    void segment();
    
private:
    /*
     * Class attributes
     */

    // Data path (frames and masks directories, and BBs/Rects files)
	string m_DataPath;
   
	// Grid parameters
    const unsigned int m_hp;
    const unsigned int m_wp;
    
    // Modalities' parametrization
	const ColorParametrization m_ColorParam;
	const MotionParametrization m_MotionParam;
    const DepthParametrization m_DepthParam;
	const ThermalParametrization m_ThermalParam;
    
    // Clustering pre-classification
    const int m_NumClusters;
	// Number of mixtures (classes) in the GMM
	const int m_NumMixtures;
    
    /*
     * Private methods
     */
    
    // Load frames of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector< vector<Rect> > & rects, vector< vector<int> > & tags);
    
    // Training data and testing data random indexing
    cv::Mat shuffled(int a, int b, RNG randGen);
    // Indexing
    void select(vector<GridMat> grids, vector<int> indices, vector<GridMat> & selection);
    // Trim subimages (using the rects provided) from frames
    void grid(vector<cv::Mat> frames, vector< vector<Rect> > boundingRects, vector< vector<int> > tags, unsigned int crows, unsigned int ccols, vector<GridMat> & grids);
    
    void segmentColor();
    void segmentMotion();
    void segmentThermal();
    void segmentDepth();

    // DEBUG
    void loadDebugTestData(const char* path, vector<cv::Mat> & test, vector<cv::Mat> & testmasks, vector< vector<Rect> > & testrects);
};

#endif /* defined(__Segmenthreetion__TrimodalSegmentator__) */
