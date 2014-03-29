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


#include "ColorFeatureExtractor.h"
#include "MotionFeatureExtractor.h"
#include "DepthFeatureExtractor.h"
#include "ThermalFeatureExtractor.h"

#include "ColorParametrization.hpp"
#include "MotionParametrization.hpp"
#include "ThermalParametrization.hpp"
#include "DepthParametrization.hpp"

#include "ModalityPrediction.h"

#include "StatTools.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#include <boost/assign/std/vector.hpp>

using namespace boost::assign;
using namespace std;

int main(int argc, const char* argv[])
{
    //
    // Parametrization
    //
    
    // Dataset handling
	// Create a reader pointing the data streams

#ifdef _WIN32
	string dataPath = "../Sequences/";
#elif __APPLE__
	string dataPath = "../../Sequences/";
#endif

	vector<string> sequences;
    sequences += "Scene1/", "Scene2/", "Scene3/";
	
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
    
	vector<int> nmixtures;
    nmixtures += 4; // classification parameter (training step)
    vector<int> nlikelicuts;
    nlikelicuts += -40, -20, -10, 0, 10, 20, 40;
    
    // Validation procedure
    
    int kTest = 10; // number of folds in the outer cross-validation
    int kModelSelec = 3;
    int seed = 74;
    
    //
    // Execution
    //
    
    ModalityReader reader(dataPath);
    reader.setMasksOffset(masksOffset);
    
//    ModalityData dData;
//    ModalityData cData;
//    ModalityData tData;
    
//    ModalityWriter writer(dataPath);
    
    // Depth
//    reader.read("Depth", dData);
    
//    DepthBackgroundSubtractor dBS(fParam);
//    dBS.setMasksOffset(masksOffset);
//    dBS.getMasks(dData);
//    dBS.getBoundingRects(dData);
//    dBS.adaptGroundTruthToReg(dData);
//    dBS.getGroundTruthBoundingRects(dData);
//    dBS.getRoiTags(dData, false);
//    
//    writer.write("Depth", dData);

    
    //Color
//    reader.read("Color", cData);
    
//    ColorBackgroundSubtractor cBS;
//    cBS.setMasksOffset(masksOffset);
//    cBS.getMasks(dData, cData);
//    cBS.getBoundingRects(dData, cData);
//    cBS.adaptGroundTruthToReg(dData, cData);
//    cBS.getGroundTruthBoundingRects(dData,cData);
//    cBS.getRoiTags(dData, cData);
//    
//    writer.write("Color", cData);
    
    // Thermal
    // <------
	//reader.read("Thermal", tData);
    
//    ThermalBackgroundSubtractor tBS;
//    tBS.setMasksOffset(masksOffset);
//    tBS.getMasks(dData, tData);
//    tBS.getBoundingRects(dData, tData);
//   // tBS.adaptGroundTruthToReg(tData);
//    tBS.getRoiTags(dData, tData);
//    
//    writer.write("Thermal", tData);

    
    // Color description

//    ModalityGridData cGridData;
//
//    ColorFeatureExtractor cFE(cParam);
//	for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Reading color frames in scene " << s << ".." << endl;
//		reader.read("Color", sequences[s], "jpg", hp, wp, cGridData);
//        cout << "Describing color..." << endl;
//		cFE.describe(cGridData);
//	}
//    
//    cGridData.saveDescription("Color.yml");

    
    //
    // Feature extraction
    //
    
    // Motion description
    
//    ModalityGridData mGridData;
//
//    MotionFeatureExtractor mFE(mParam);
//    for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Computing motion (from read color) frames in scene " << s << ".." << endl;
//		reader.read("Motion", sequences[s], "jpg", hp, wp, mGridData);
//        cout << "Describing motion..." << endl;
//        mFE.describe(mGridData);
//	}
//
//	mGridData.saveDescription("Motion.yml");

    
    // Depth description
    
//	ModalityGridData dGridData;
//    DepthFeatureExtractor dFE(dParam);
//
//	for (int s = 0; s < sequences.size(); s++)
//	{
//		reader.read("Depth", sequences[s], "png", hp, wp, dGridData);
//
//		dFE.describe(dGridData);
//	}
//
//	dGridData.saveDescription("Depth.yml");

    
    // Thermal description
    
//    ModalityGridData tGridData;
//
//    ThermalFeatureExtractor tFE(tParam);
//	for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Reading thermal frames in scene " << s << ".." << endl;
//		reader.read("Thermal", sequences[s], "jpg", hp, wp, tGridData);
//        cout << "Describing thermal..." << endl;
//		tFE.describe(tGridData);
//	}
//    
//    tGridData.saveDescription("Thermal.yml");

    ModalityGridData tMockData;
    reader.mockread("Thermal", sequences, "jpg", hp, wp, tMockData); // mockread :D
    ModalityGridData cMockData; // TODO: try with Motion
    reader.mockread("Color", sequences, "jpg", hp, wp, cMockData); // mockread :D
    
    tMockData.loadDescription("Thermal.yml");
    cMockData.loadDescription("Color.yml");
    
    // Important piece of code
    vector<ModalityGridData*> mgds;
    mgds += &tMockData, &cMockData;
    reader.agreement(mgds);
    
    GridMat tPredictions, tLoglikelihoods;
    
    ModalityPrediction<cv::EM> tPrediction;
    tPrediction.setData(tMockData);

    tPrediction.setNumOfMixtures(nmixtures);
    tPrediction.setLoglikelihoodThresholds(nlikelicuts);

    tPrediction.setModelValidation(kTest, seed);
    tPrediction.setModelSelection(kModelSelec, true);
    
    tPrediction.predict(tPredictions, tLoglikelihoods);

    return 0;
}

