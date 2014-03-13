//
//  main.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "ModalityReader.h"
#include "ModalityWriter.h"
#include "ModalityData.hpp"
#include "ModalityGridData.hpp"
#include "GridPartitioner.h"

#include "ForegroundParametrization.hpp"
#include "DepthBackgroundSubtractor.h"
#include "ColorBackgroundSubtractor.h"
#include "ThermalBackgroundSubtractor.h"

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
    
    //Background subtraction
    
    ForegroundParametrization fParam;
    
    int nftl[] = {35,200,80}; //frames needed to learn the background models for each sequence
    const std::vector<int> nFramesToLearn(nftl, nftl + 3);
    
    fParam.numFramesToLearn = nFramesToLearn;
    fParam.boundingBoxMinArea = 0.001;
    fParam.otsuMinArea = 0.02;
    fParam.otsuMinVariance1 = 8.3;
    fParam.otsuMinVariance2 = 12;
    
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
    
    ModalityData dData;
    ModalityData dGridData;
    
    ModalityData cData;
    ModalityData cGridData;

    
    ModalityData tData;
    ModalityGridData tGridData;
    
    
    ModalityReader reader(dataPath);
    reader.setMasksOffset(masksOffset);
    
    
    ModalityWriter writer(dataPath);
    
    // Depth
    reader.read("Depth", dData);
    
    DepthBackgroundSubtractor dBS(fParam);
    dBS.setMasksOffset(masksOffset);
    dBS.getMasks(dData);
    dBS.getBoundingRects(dData);
    dBS.adaptGroundTruthToReg(dData);
    dBS.getGroundTruthBoundingRects(dData);
    dBS.getRoiTags(dData, true);
    
    writer.write("Depth", dData);
    
    //Color
    reader.read("Color", cData);
    
    ColorBackgroundSubtractor cBS;
    cBS.setMasksOffset(masksOffset);
    cBS.getMasks(cData, dData);
    cBS.getBoundingRects(cData, dData);
    cBS.adaptGroundTruthToReg(cData, dData);
    cBS.getGroundTruthBoundingRects(cData,dData);
    cBS.getRoiTags(cData, dData);
    
    // Thermal
    // <------
    reader.read("Thermal", tData);
    
    
    ThermalBackgroundSubtractor tBS;
    tBS.setMasksOffset(masksOffset);
    tBS.getMasks(tData, dData);
    tBS.getBoundingRects(tData, dData);
    tBS.adaptGroundTruthToReg(tData);

    GridPartitioner partitioner;
    partitioner.setGridPartitions(hp, wp);
    partitioner.grid(tData, tGridData); // perform "gridding"
    
  /*  GridMat tDescriptors;
    ThermalFeatureExtractor tFE(tParam);
    tFE.describe(tGridData, tDescriptors); // perform description
    */
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

