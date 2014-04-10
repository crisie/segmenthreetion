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

#include "FusionPrediction.h"

#include "GridMapWriter.h"

#include "StatTools.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#include <boost/assign/std/vector.hpp>

#include <matio.h>

using namespace boost::assign;
using namespace std;

int main(int argc, const char* argv[])
{ 

// =============================================================================
//  Parametrization
// =============================================================================
    
    // Dataset handling, create a reader pointing the data streams
    
#ifdef _WIN32 // Visual Studio
	string dataPath = "../Sequences/";
#elif __APPLE__ // Xcode
	string dataPath = "../../Sequences/";
#endif

	vector<string> sequences;
    sequences += "Scene1/", "Scene2/", "Scene3/";
	
	const unsigned char masksOffset = 200;
    
	// Background subtraction parametrization
    
    ForegroundParametrization fParam;
    
    int nftl[] = {35,200,80}; //frames needed to learn the background models for each sequence
    const std::vector<int> nFramesToLearn(nftl, nftl + 3);
    
    fParam.numFramesToLearn = nFramesToLearn;
    fParam.boundingBoxMinArea = 0.001;
    fParam.otsuMinArea = 0.02;
    fParam.otsuMinVariance1 = 8.3;
    fParam.otsuMinVariance2 = 12;
    
    // Feature extraction parametrization
    
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
    dParam.normalsRadius    = 0.04;
    
    ThermalParametrization tParam;
    tParam.ibins    = 8;
    tParam.oribins  = 8;
    
    // Leraning algorithms' parametrization
    
    vector<float> tmp;
    
    // all modalities, except for ramanan
	vector<int> nmixtures;
    nmixtures += 2, 3, 4, 5, 7, 9, 11; // classification parameter (training step)
    tmp = cvx::linspace(-2, 2, 22); // -1, -0.8, -0.6, ..., 0.8
    vector<float> likelicuts (tmp.begin()+1, tmp.end()-1);
    
    // ramanan
    tmp = cvx::linspace(0, 0.15, 20);
    vector<float> ratios (tmp.begin()+1, tmp.end()-1);
    tmp = cvx::linspace(0, 1.00, 20);
    vector<float> scoreThresholds (tmp.begin()+1, tmp.end()-1);
    
    double colorVariance = 0.99; // variance to keep in PCA's dim reduction in ColorModality
    
    // Validation procedure
    
    int kTest = 10; // number of folds in the outer cross-validation
    int kModelSelec = 9;
    int seed = 74;
 
    
// =============================================================================
//  Execution
// =============================================================================
    
    ModalityReader reader(dataPath);
    reader.setMasksOffset(masksOffset);
    
    
    //
    // Background subtraction
    //
    
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

    
    //
    // Feature extraction
    //

//    // Color description
//
//    ModalityGridData cGridData;
//
//    ColorFeatureExtractor cFE(cParam);
//	for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Reading color frames in scene " << s << ".." << endl;
//		reader.read("Color", sequences[s], "jpg", hp, wp, cGridData);
//        cout << "Describing color..." << endl;
//		cFE.describe(cGridData);
//        cGridData.saveDescription(dataPath, sequences[s], "Color.yml");
//    }
//
//    // Motion description
//    
//    ModalityGridData mGridData;
//
//    MotionFeatureExtractor mFE(mParam);
//    for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Computing motion (from read color) frames in scene " << s << ".." << endl;
//		reader.read("Motion", sequences[s], "jpg", hp, wp, mGridData);
//        cout << "Describing motion..." << endl;
//        mFE.describe(mGridData);
//        mGridData.saveDescription(dataPath, sequences[s], "Motion.yml");
//	}
//    
//    // Depth description
//    
////    ModalityGridData dGridData;
////    
////    DepthFeatureExtractor dFE(dParam);
////	for (int s = 0; s < sequences.size(); s++)
////	{
////        cout << "Reading depth frames in scene " << s << ".." << endl;
////		reader.read("Depth", sequences[s], "png", hp, wp, dGridData);
////        cout << "Describing depth..." << endl;
////		dFE.describe(dGridData);
////        dGridData.saveDescription(dataPath, sequences[s], "Depth.yml");
////	}
//    
//    // Thermal description
//    
//    ModalityGridData tGridData;
//
//    ThermalFeatureExtractor tFE(tParam);
//	for (int s = 0; s < sequences.size(); s++)
//	{
//        cout << "Reading thermal frames in scene " << s << ".." << endl;
//		reader.read("Thermal", sequences[s], "jpg", hp, wp, tGridData);
//        cout << "Describing thermal..." << endl;
//		tFE.describe(tGridData);
//        tGridData.saveDescription(dataPath, sequences[s], "Thermal.yml");
//	}
//
//    cGridData.clear();
//    mGridData.clear();
////    dGridData.clear();
//    tGridData.clear();
    
    
    //
    // "Light" data re-loading
    //
    
    ModalityGridData mMockData, dMockData, tMockData, cMockData;
    
    reader.mockread("Color", sequences, "jpg", hp, wp, cMockData);
    reader.mockread("Motion", sequences, "jpg", hp, wp, mMockData);
    reader.mockread("Depth", sequences, "png", hp, wp, dMockData);
    reader.mockread("Thermal", sequences, "jpg", hp, wp, tMockData);

//    cMockData.loadDescription(dataPath, sequences, "Color.yml");
//    mMockData.loadDescription(dataPath, sequences, "Motion.yml");
//    dMockData.loadDescription(dataPath, sequences, "Depth.yml");
//    tMockData.loadDescription(dataPath, sequences, "Thermal.yml");
    
    
    //
    // Individual prediction
    //
    
    GridMat mPredictions, dPredictions, tPredictions, cPredictions, rPredictions;
    GridMat mLoglikelihoods, dLoglikelihoods, tLoglikelihoods, cLoglikelihoods, rScores;
    GridMat mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin, rDistsToMargin;
    GridMat mAccuracies, dAccuracies, tAccuracies, cAccuracies;
    
//    // Ramanan
//
    
    ModalityGridData rGridData;
    reader.read("Ramanan", sequences, "mat", hp, wp, rGridData);
    
    ModalityPrediction<cv::Mat> rprediction;
    
    rprediction.setData(rGridData);
    
    rprediction.setPositiveClassificationRatios(ratios);
    rprediction.setScoreThresholds(scoreThresholds);
    
    rprediction.setValidationParameters(kTest, seed);
    rprediction.setModelSelection(false); // false load it from disk (see .h)
    rprediction.setModelSelectionParameters(kModelSelec, true);
    
    rprediction.compute(rPredictions, rScores, rDistsToMargin);
    
    rPredictions.save("rPredictions.yml");
    rScores.save("rScores.yml");
    rDistsToMargin.save("rDistsToMargin.yml");
    
    rGridData.clear();

//    // Other modalitites
    
    ModalityPrediction<cv::EM> prediction;
//
//    prediction.setNumOfMixtures(nmixtures);
//    prediction.setLoglikelihoodThresholds(likelicuts);
//
//    prediction.setValidationParameters(kTest, seed);
//    prediction.setModelSelection(false); // false load it from disk (see .h)
//    prediction.setModelSelectionParameters(kModelSelec, true);
//    
//    // Motion
//    prediction.setData(mMockData);
//    
//    prediction.compute(mPredictions, mLoglikelihoods, mDistsToMargin, mAccuracies);
//
//    mPredictions.save("mPredictions.yml");
//    mLoglikelihoods.save("mLoglikelihoods.yml");
//    mDistsToMargin.save("mDistsToMargin.yml");
//    mAccuracies.save("mAccuracies.yml");
//    
//    // Depth
//    prediction.setData(dMockData);
//
//    prediction.compute(dPredictions, dLoglikelihoods, dDistsToMargin, dAccuracies);
//    
//    dPredictions.save("dPredictions.yml");
//    dLoglikelihoods.save("dLoglikelihoods.yml");
//    dDistsToMargin.save("dDistsToMargin.yml");
//    dAccuracies.save("dAccuracies.yml");
//    
//     // Thermal
//    prediction.setData(tMockData);
//
//    prediction.compute(tPredictions, tLoglikelihoods, tDistsToMargin, tAccuracies);
//
//    tPredictions.save("tPredictions.yml");
//    tLoglikelihoods.save("tLoglikelihoods.yml");
//    tDistsToMargin.save("tDistsToMargin.yml");
//    tAccuracies.save("tAccuracies.yml");
//    
//    // Color
//    prediction.setData(cMockData);
//    prediction.setDimensionalityReduction(colorVariance);
//
//    prediction.compute(cPredictions, cLoglikelihoods, cDistsToMargin, cAccuracies);
//    
//    cPredictions.save("cPredictions.yml");
//    cLoglikelihoods.save("cLoglikelihoods.yml");
//    cDistsToMargin.save("cDistsToMargin.yml");
//    cAccuracies.save("cAccuracies.yml");

//
//    // DEBUG: study the distributions of loglikelihoods in a certain modality
//    cv::Mat l, sbjDist, objDist;
//    sbjDist.setTo(0);
//    objDist.setTo(0);
//    prediction.computeLoglikelihoodsDistribution(60, -20, 20, sbjDist, objDist);
//    cvx::linspace(-20, 20, 60, l);
//    cout << l << endl;
//    cout << sbjDist << endl;
//    cout << objDist << endl;
    

    //
    // Fusion
    //
    
    mPredictions.load("mPredictions.yml");
    dPredictions.load("dPredictions.yml");
    tPredictions.load("tPredictions.yml");
    cPredictions.load("cPredictions.yml");
    rPredictions.load("rPredictions.yml");
    
    mLoglikelihoods.load("mLoglikelihoods.yml");
    dLoglikelihoods.load("dLoglikelihoods.yml");
    tLoglikelihoods.load("tLoglikelihoods.yml");
    cLoglikelihoods.load("cLoglikelihoods.yml");
    rScores.load("rScores.yml");
    
    mDistsToMargin.load("mDistsToMargin.yml");
    dDistsToMargin.load("dDistsToMargin.yml");
    tDistsToMargin.load("tDistsToMargin.yml");
    cDistsToMargin.load("cDistsToMargin.yml");
    rDistsToMargin.load("rDistsToMargin.yml");
    
    cout << "Computing individual predictions cells consensus... " << endl;
    
    GridMat mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
            cConsensusPredictions, rConsensusPredictions;
    
    cout << "... motion (hoof) modality" << endl;
    prediction.computeGridPredictionsConsensus(mMockData, mPredictions, mDistsToMargin, mConsensusPredictions);
    cout << "Motion modality: " << accuracy(mMockData.getTagsMat(), mPredictions) << endl;
    cout << "Motion modality (c): " << accuracy(mMockData.getTagsMat(), mConsensusPredictions) << endl;
    
    cout << "... color (hon) modality" << endl;
    prediction.computeGridPredictionsConsensus(dMockData, dPredictions, dDistsToMargin, dConsensusPredictions);
    cout << "Depth modality: " << accuracy(dMockData.getTagsMat(), dPredictions) << endl;
    cout << "Depth modality (c): " << accuracy(dMockData.getTagsMat(), dConsensusPredictions) << endl;
    
    cout << "... thermal (hiog) modality" << endl;
    prediction.computeGridPredictionsConsensus(tMockData, tPredictions, tDistsToMargin, tConsensusPredictions);
    cout << "Thermal modality: " << accuracy(tMockData.getTagsMat(), tPredictions) << endl;
    cout << "Thermal modality (c): " << accuracy(tMockData.getTagsMat(), tConsensusPredictions) << endl;
    
    cout << "... color (hog) modality" << endl;
    prediction.computeGridPredictionsConsensus(cMockData, cPredictions, cDistsToMargin, cConsensusPredictions);
    cout << "Color modality: " << accuracy(cMockData.getTagsMat(), cPredictions) << endl;
    cout << "Color modality (c): " << accuracy(cMockData.getTagsMat(), cConsensusPredictions) << endl;
    
    cout << "... ramanan (ramanan scores) modality" << endl;
    prediction.computeGridPredictionsConsensus(cMockData, rPredictions, rDistsToMargin, rConsensusPredictions);
    cout << "Ramanan modality: " << accuracy(cMockData.getTagsMat(), rPredictions) << endl;
    cout << "Ramanan modality (c): " << accuracy(cMockData.getTagsMat(), rConsensusPredictions) << endl;
    
    
    cout << "Computing fusion predictions ... " << endl;
    
    vector<GridMat> predictions, loglikelihoods, distsToMargin; // put together all the data
    predictions     += mPredictions, dPredictions, tPredictions, cPredictions, rPredictions;
    loglikelihoods  += mLoglikelihoods, dLoglikelihoods, tLoglikelihoods, cLoglikelihoods, rScores;
    distsToMargin   += mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin, rDistsToMargin;
    
    // Simple fusion
    
    cout << "... naive approach" << endl;
    
    GridMat simpleFusionPredictions1; // Cells' pre-consensued predictions
    GridMat simpleFusionPredictions2; // Cells' post-consensued predictions
    GridMat simpleFusionPredictions3; // Cells' distances to margin
    
    SimpleFusionPrediction<cv::EM> simpleFusion;
    
    // Indiviual modalities' grid consensus first, then fusion
    vector<GridMat> consensuedPredictions;
    consensuedPredictions += mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
            cConsensusPredictions, rConsensusPredictions;
    simpleFusion.compute(consensuedPredictions, distsToMargin, simpleFusionPredictions1);
    cout << "Cells' pre-consensued predictions: " << accuracy(cMockData.getTagsMat(), simpleFusionPredictions1) << endl;
    
    // Fusion, and then grid consensus
    GridMat simpleFusionDistsToMargin2;
    simpleFusion.compute(predictions, distsToMargin, simpleFusionPredictions2, simpleFusionDistsToMargin2);
    prediction.computeGridPredictionsConsensus(cMockData, simpleFusionPredictions2, simpleFusionDistsToMargin2, simpleFusionPredictions2);
    cout << "Cells' post-consensued predictions: " << accuracy(cMockData.getTagsMat(), simpleFusionPredictions2) << endl;
    
    GridMat simpleFusionDistsToMargin3;
    simpleFusion.compute(distsToMargin, simpleFusionPredictions3, simpleFusionDistsToMargin3);
    prediction.computeGridPredictionsConsensus(cMockData, simpleFusionPredictions3, simpleFusionDistsToMargin3, simpleFusionPredictions3);
    cout << "Cells' distances to margin: " << accuracy(cMockData.getTagsMat(), simpleFusionPredictions3) << endl;
    
    
    // SVM fusion
    
    cout << "... SVM approach" << endl;
    
    vector<float> cs, gammas;
    cs += 1e-21, 1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4; // example
    gammas += 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1, 1e1, 1e2;
    
    GridMat svmFusionPredictions1, svmFusionPredictions2,
            svmFusionPredictions3, svmFusionPredictions4;
    
    ClassifierFusionPrediction<cv::EM,CvSVM> svmFusion;
    
    svmFusion.setData(distsToMargin, consensuedPredictions);
    svmFusion.setResponses(cMockData.getTagsMat());
    
    svmFusion.setModelSelection(true);
    svmFusion.setValidationParameters(kTest, seed);
    svmFusion.setModelSelectionParameters(kModelSelec, true);

    // Linear-kernel SVM
    svmFusion.setKernelType(CvSVM::LINEAR);
    svmFusion.setCs(cs);
    
    svmFusion.setModelSelection(false);
    svmFusion.setStackedPrediction(false);
    svmFusion.compute(svmFusionPredictions1);
    cout << "SVM fusion linear: " << accuracy(cMockData.getTagsMat(), svmFusionPredictions1) << endl;

    svmFusion.setModelSelection(true);
    svmFusion.setStackedPrediction(true);
    svmFusion.compute(svmFusionPredictions2);
    cout << "SVM fusion linear /w preds: " << accuracy(cMockData.getTagsMat(), svmFusionPredictions2) << endl;
    
    // RBF-kernel SVM
    svmFusion.setKernelType(CvSVM::RBF);
    svmFusion.setGammas(gammas);
    
    svmFusion.setStackedPrediction(false);
    svmFusion.compute(svmFusionPredictions3);
    cout << "SVM fusion rbf: " << accuracy(cMockData.getTagsMat(), svmFusionPredictions3) << endl;
    
    svmFusion.setStackedPrediction(true);
    svmFusion.compute(svmFusionPredictions4);
    cout << "SVM fusion rbf /w preds: " << accuracy(cMockData.getTagsMat(), svmFusionPredictions4) << endl;
    
    
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

