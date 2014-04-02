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

#include "GridMapWriter.h"

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
    
    //
	//Background subtraction
    //
    
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
    cParam.hogbins = 288; // total feature vector length
    
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
    nmixtures += 3, 6; // classification parameter (training step)
    vector<int> nlikelicuts;
    nlikelicuts += 0, 40;
    
    // Validation procedure
    
    int kTest = 3; // number of folds in the outer cross-validation
    int kModelSelec = 2;
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

    ModalityGridData cGridData;

    ColorFeatureExtractor cFE(cParam);
	for (int s = 0; s < sequences.size(); s++)
	{
        cout << "Reading color frames in scene " << s << ".." << endl;
		reader.read("Color", sequences[s], "jpg", hp, wp, cGridData);
        cout << "Describing color..." << endl;
		cFE.describe(cGridData);
        cGridData.saveDescription(dataPath, sequences[s], "Color.yml");
    }

    // Motion description
    
    ModalityGridData mGridData;

    MotionFeatureExtractor mFE(mParam);
    for (int s = 0; s < sequences.size(); s++)
	{
        cout << "Computing motion (from read color) frames in scene " << s << ".." << endl;
		reader.read("Motion", sequences[s], "jpg", hp, wp, mGridData);
        cout << "Describing motion..." << endl;
        mFE.describe(mGridData);
        mGridData.saveDescription(dataPath, sequences[s], "Motion.yml");
	}
    
    // Depth description
    
    ModalityGridData dGridData;
    
    DepthFeatureExtractor dFE(dParam);
	for (int s = 0; s < sequences.size(); s++)
	{
        cout << "Reading depth frames in scene " << s << ".." << endl;
		reader.read("Depth", sequences[s], "png", hp, wp, dGridData);
        cout << "Describing depth..." << endl;
		dFE.describe(dGridData);
        dGridData.saveDescription(dataPath, sequences[s], "Depth.yml");
	}
    
    // Thermal description
    
    ModalityGridData tGridData;

    ThermalFeatureExtractor tFE(tParam);
	for (int s = 0; s < sequences.size(); s++)
	{
        cout << "Reading thermal frames in scene " << s << ".." << endl;
		reader.read("Thermal", sequences[s], "jpg", hp, wp, tGridData);
        cout << "Describing thermal..." << endl;
		tFE.describe(tGridData);
        tGridData.saveDescription(dataPath, sequences[s], "Thermal.yml");
	}
//
//    cGridData.clear();
//    mGridData.clear();
////    dGridData.clear();
//    tGridData.clear();
//    
//    
//    //
//    // Data re-loading
//    //
//    
//    ModalityGridData cMockData;
//    reader.mockread("Color", sequences, "jpg", hp, wp, cMockData);
//    ModalityGridData mMockData;
//    reader.mockread("Motion", sequences, "jpg", hp, wp, mMockData);
////    ModalityGridData dMockData;
////    reader.mockread("Depth", sequences, "png", hp, wp, dMockData);
//    ModalityGridData tMockData;
//    reader.mockread("Thermal", sequences, "jpg", hp, wp, tMockData);
//
//    cMockData.loadDescription(dataPath, sequences, "Color.yml");
//    mMockData.loadDescription(dataPath, sequences, "Motion.yml");
////    dMockData.loadDescription(dataPath, sequences, "Depth.yml");
//    tMockData.loadDescription(dataPath, sequences, "Thermal.yml");
//    
//    //
//    // Prediction
//    //
//    
//    ModalityPrediction<cv::EM> prediction;
//
//    prediction.setNumOfMixtures(nmixtures);
//    prediction.setLoglikelihoodThresholds(nlikelicuts);
//
//    prediction.setModelValidation(kTest, seed);
//    prediction.setModelSelection(kModelSelec, true);
//    
//    // Thermal
//    prediction.setData(tMockData);
//
//    GridMat tPredictions, tLoglikelihoods;
//    prediction.compute(tPredictions, tLoglikelihoods);
//    
//    tPredictions.save("tPredictions.yml");
//    tLoglikelihoods.save("tLoglikelihoods.yml");
//
//    // Color
//    prediction.setData(cMockData);
//
//    GridMat cPredictions, cLoglikelihoods;
//    prediction.compute(cPredictions, cLoglikelihoods);
//
//    cPredictions.save("cPredictions.yml");
//    cLoglikelihoods.save("cLoglikelihoods.yml");
//    
//    
//    //
//    // Map writing
//    //
//       
//    tPredictions.load("tPredictions.yml");
//    
//    GridMapWriter mapWriter;
//    
//    mapWriter.write<unsigned char>(tMockData, tPredictions, "Predictions/");
//    
////    GridMat tNormLoglikelihoods;
////    tLoglikelihoods.normalize(tNormLoglikelihoods);
////    mapWriter.write<float>(tMockData, tNormLoglikelihoods, "Thermal/Loglikelihoods/"); // normalized values are float

    return 0;
}

