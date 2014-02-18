//
//  TrimodalSegmentator.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__TrimodalSegmentator__
#define __Segmenthreetion__TrimodalSegmentator__

#include "FeatureExtractor.h"
#include "ColorParametrization.hpp"
#include "MotionParametrization.hpp"
#include "DepthParametrization.hpp"
#include "ThermalParametrization.hpp"

#include "GridMat.h"
#include "TrimodalClusterer.h"

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TrimodalSegmentator
{
public:
    /*
     * Constructors
     */
    TrimodalSegmentator(const unsigned char offsetID);
    
    /*
     * Public methods
     */
    
    // Set the path containing the sequence in which the segmentation will be performed
    void setDataPath(string dataPath);
    
    // Run the program
    void extractFeatures(const unsigned int hp, const unsigned int wp,
                         const ColorParametrization cParam, const MotionParametrization mParam, const DepthParametrization dParam, const ThermalParametrization tParam);
    
private:
    /*
     * Class attributes
     */

    // Data path (frames and masks directories, and BBs/Rects files)
	string m_DataPath;
    vector<string> m_ScenesPaths;
   
    const unsigned char m_OffsetID;
    
	// Grid parameters
    unsigned int m_hp;
    unsigned int m_wp;
    
    // Modalities' parametrization
	const ColorParametrization m_ColorParam;
	const MotionParametrization m_MotionParam;
    const DepthParametrization m_DepthParam;
	const ThermalParametrization m_ThermalParam;
    
	// Number of mixtures (classes) in the GMM
	unsigned int m_NumMixtures;
    
    /*
     * Private methods
     */
    
    //
    // Data handling
    //
    
    // Load frames of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector< vector<cv::Rect> > & rects, vector< vector<int> > & tags);
    // Trim subimages (using the rects provided) from frames
    void grid(vector<cv::Mat> frames, vector< vector<cv::Rect> > boundingRects, vector< vector<int> > tags, unsigned int crows, unsigned int ccols, vector<GridMat> & grids);
    
    //
    // Feature extraction
    //
    
    void extractModalityFeatures(string scenePath, string modality, FeatureExtractor* fe,
                                 GridMat& subDescriptors, GridMat& objDescriptors, GridMat& unkDescriptors);

    //
    // Auxiliary methods
    //
    
    // Training data and testing data random indexing
    cv::Mat shuffled(int a, int b, cv::RNG randGen);
    // Indexing
    void select(vector<GridMat> grids, vector<int> indices, vector<GridMat> & selection);

    //
    // *** Debug ***
    //
    
    void loadDebugTestData(const char* path, vector<cv::Mat> & test, vector<cv::Mat> & testmasks, vector< vector<cv::Rect> > & testrects);
};

#endif /* defined(__Segmenthreetion__TrimodalSegmentator__) */
