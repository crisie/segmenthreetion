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
    TrimodalSegmentator(const unsigned int hp, const unsigned int wp, const unsigned char offsetID);
    
    /*
     * Public methods
     */
    
    // Data handling
    
    void setDataPath(string dataPath);
    void getModalityData(std::string modality, std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, cv::Mat& tags);
    
    // Cell feature extraction
    
    void extractColorFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const ColorParametrization param, GridMat& descriptors);
    void extractMotionFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const MotionParametrization param, GridMat& descriptors);
    void extractDepthFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const DepthParametrization param, GridMat& descriptors);
    void extractThermalFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const ThermalParametrization param, GridMat& descriptors);
    
    // Cell classification
    
    void computeLoglikelihoods(GridMat descriptors, cv::Mat tags, GridMat& loglikelihoods);
    
    
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
    
    // Data handling
    
    // Load frames of a modality within a directory
    void loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames);
    // Load people bounding boxes (rects)
    void loadBoundingRects(string file, vector< vector<cv::Rect> > & rects, vector< vector<int> > & tags);
    // Trim subimages (using the rects provided) from frames
    void grid(vector<cv::Mat>& frames, vector< vector<cv::Rect> > rects, unsigned int crows, unsigned int ccols, vector<GridMat>& grids);
    void grid(vector<cv::Mat>& frames, vector< vector<cv::Rect> > rects, vector< vector<int> > rtags, unsigned int crows, unsigned int ccols, vector<GridMat>& grids, cv::Mat& tags);
    
    // Feature extraction
    // ...
    
    
    // Cell classification
    // ...
    
//    void modelPredictor();
//    void predict();
    
    
    // Auxiliary methods
    
    // Training data and testing data random indexing
    cv::Mat shuffled(int a, int b, cv::RNG randGen);
    // Indexing
    void select(vector<GridMat> grids, vector<int> indices, vector<GridMat> & selection);
};

#endif /* defined(__Segmenthreetion__TrimodalSegmentator__) */
