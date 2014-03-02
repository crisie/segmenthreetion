//
//  main.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "ModalityReader.h"
#include "ModalityData.hpp"
#include "ModalityGridData.hpp"
#include "GridPartitioner.h"

#include "ThermalFeatureExtractor.h"

#include "ColorParametrization.hpp"
#include "MotionParametrization.hpp"
#include "ThermalParametrization.hpp"
#include "DepthParametrization.hpp"

#include "ModalityPrediction.h"

#include "StatTools.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;

int main(int argc, const char* argv[])
{
    //
    // Parametrization
    //
    
    // Dataset handling
    
    string dataPath = "../../Sequences/";
    const unsigned char masksOffset = 200;
    
    // Feature extraction
    
    const unsigned int hp = 2; // partitions in height
    const unsigned int wp = 2; // partitions in width
    
    ColorParametrization cParam;
    cParam.winSizeX = 64;
    cParam.winSizeY = 128;
    cParam.blockSizeX = 32;
    cParam.blockSizeY = 32;
    cParam.cellSizeX = 16;
    cParam.cellSizeY = 16;
    cParam.nbins = 9;
    
    MotionParametrization mParam;
    mParam.hoofbins = 8;
    mParam.pyr_scale = 0.5;
    mParam.levels = 3;
    mParam.winsize = 15;
    mParam.iterations = 3;
    mParam.poly_n = 5;
    mParam.poly_sigma = 1.2;
    mParam.flags = 0;
    
    DepthParametrization dParam;
    dParam.thetaBins        = 8;
    dParam.phiBins          = 8;
    dParam.normalsRadius    = 0.02;
    
    ThermalParametrization tParam;
    tParam.ibins    = 8;
    tParam.oribins  = 8;
    
    // Classification step
    
	int numMixtures[] = {2,3,4,5,6}; // classification parameter (training step)
    
    // Validation procedure
    
    int kTest = 10; // number of folds in the outer cross-validation
    int kModelSelec = 3;
    int seed = 74;
    
    //
    // Execution
    //
    
    ModalityData tData;
    ModalityGridData tGridData;
    
    ModalityReader reader(dataPath);
    reader.setMasksOffset(masksOffset);
    
    // Thermal
    // <------
    reader.read("Thermal", tData);
    
    GridPartitioner partitioner;
    partitioner.setGridPartitions(hp, wp);
    partitioner.grid(tData, tGridData); // perform "gridding"
    
    GridMat tDescriptors;
    ThermalFeatureExtractor tFE(tParam);
    tFE.describe(tGridData, tDescriptors); // perform description
    
//    GridMat tLoglikelihoods;
    
    ModalityPrediction<cv::EM> tPrediction;
//    tPredictor.setModelSelection(kModelSelec, expand(nmixtures, nlikelicuts));
//    tPredictor.setModelSelection(kTest);
//    
//    tPredictor.setData(t)
//    tPredictor.computeLoglikelihoods(tLoglikelihoods);
    // ------>
    

    return 0;
}

