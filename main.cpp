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

#include "Validation.h"

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
    nmixtures += 2, 3, 4, 5, 6, 7, 8, 9, 11, 13; // classification parameter (training step)
    tmp = cvx::linspace(-2, 2, 32); // -1, -0.8, -0.6, ..., 0.8
    vector<float> likelicuts (tmp.begin()+1, tmp.end()-1);
    
    // ramanan
    tmp = cvx::linspace(0, 1.0, 30);
    vector<float> ratios (tmp.begin()+1, tmp.end()-1);
    tmp = cvx::linspace(0, 1.0, 30);
    vector<float> scoreThresholds (tmp.begin()+1, tmp.end()-1);
    
    double colorVariance = 0.99; // variance to keep in PCA's dim reduction in ColorModality
    
    // fusion strategies

    // ... svm params
    vector<float> cs, gammas;
    cs += 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4;
    gammas += 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2;
    
    // ... boost params
    
    vector<float> numOfWeaks, weightTrimRates;
    numOfWeaks += 10, 20, 50, 100, 200, 500, 1000;
    weightTrimRates += 0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99;
    
    // ... mlp params
    
    vector<float> hiddenLayerSizes; //, decayWeights, momentWeights;
    hiddenLayerSizes += 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50;
    //numOfWeaks += 10, 20, 50, 100, 200, 500, 1000;
    //weightTrimRates += 0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99;
    
    
    // Validation procedure
    
    int kTest = 10; // number of folds in the outer cross-validation
    int kModelSelec = 3;
    int seed = 74;
 
    // Overlap params
    vector<float> dontCareRange;
    dontCareRange += 1, 3, 5, 7, 9, 11, 13, 15, 17;
    
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
    // Individual prediction
    //
    
    GridMat mPredictions, dPredictions, tPredictions, cPredictions, rPredictions;
    GridMat mLoglikelihoods, dLoglikelihoods, tLoglikelihoods, cLoglikelihoods, rScores;
    GridMat mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin, rDistsToMargin;
    GridMat mAccuracies, dAccuracies, tAccuracies, cAccuracies, rAccuracies;
    
    GridMat aux;
    
    // Ramanan
    
//    ModalityGridData rGridData;
//    reader.read("Ramanan", sequences, "mat", hp, wp, rGridData);
//    
//    ModalityPrediction<cv::Mat> rprediction;
//    
//    rprediction.setData(rGridData);
//    
//    rprediction.setPositiveClassificationRatios(ratios);
//    rprediction.setScoreThresholds(scoreThresholds);
//    
//    rprediction.setValidationParameters(kTest, seed);
//    rprediction.setModelSelection(false); // false load it from disk (see .h)
//    rprediction.setModelSelectionParameters(kModelSelec, true);
//    
//    rprediction.compute(rPredictions, rScores, rDistsToMargin, rAccuracies);
//    
//    rPredictions.save("rPredictions.yml");
//    rScores.save("rScores.yml");
//    rDistsToMargin.save("rDistsToMargin.yml");
//    rAccuracies.save("rAccuracies.yml");
//    
//    rGridData.clear();

    // Other modalitites

    ModalityGridData mMockData, dMockData, tMockData, cMockData;
    
    reader.mockread("Color", sequences, "jpg", hp, wp, cMockData);
    reader.mockread("Motion", sequences, "jpg", hp, wp, mMockData);
    reader.mockread("Depth", sequences, "png", hp, wp, dMockData);
    reader.mockread("Thermal", sequences, "jpg", hp, wp, tMockData);
/*
    cMockData.loadDescription(dataPath, sequences, "Color.yml");
    mMockData.loadDescription(dataPath, sequences, "Motion.yml");
    dMockData.loadDescription(dataPath, sequences, "Depth.yml");
    tMockData.loadDescription(dataPath, sequences, "Thermal.yml");
    
    ModalityPrediction<cv::EM> prediction;

    prediction.setNumOfMixtures(nmixtures);
    prediction.setLoglikelihoodThresholds(likelicuts);

    prediction.setValidationParameters(kTest, seed);

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
//    
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
/*
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
    
    mAccuracies.load("mAccuracies.yml");
    dAccuracies.load("dAccuracies.yml");
    tAccuracies.load("tAccuracies.yml");
    cAccuracies.load("cAccuracies.yml");
    rAccuracies.load("rAccuracies.yml");
    
    cv::Mat means, confs;
    
    cout << "Motion modality:" << endl;
    computeConfidenceInterval(mAccuracies, means, confs);
    cout << means << endl;
    cout << confs << endl;
    
    cout << "Depth modality:" << endl;
    computeConfidenceInterval(dAccuracies, means, confs);
    cout << means << endl;
    cout << confs << endl;
    
    cout << "Thermal modality:" << endl;
    computeConfidenceInterval(tAccuracies, means, confs);
    cout << means << endl;
    cout << confs << endl;
    
    cout << "Color modality:" << endl;
    computeConfidenceInterval(cAccuracies, means, confs);
    cout << means << endl;
    cout << confs << endl;
    
    cout << "Ramanan modality:" << endl;
    computeConfidenceInterval(rAccuracies, means, confs);
    cout << means << endl;
    cout << confs << endl;

*/
    cout << "Computing individual predictions cells consensus... " << endl;

    cv::Mat mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
            cConsensusPredictions, rConsensusPredictions;
    cv::Mat mConsensusDistsToMargin, dConsensusDistsToMargin, tConsensusDistsToMargin,
            cConsensusDistsToMargin, rConsensusDistsToMargin;

    cv::Mat partitions;
  /*
   cvpartition(cMockData.getTagsMat(), kTest, seed, partitions);
    cv::Mat accuracies;
    float mean, conf;
    
    cout << "... motion (hoof) modality" << endl;
    prediction.computeGridPredictionsConsensus(mMockData, mPredictions, mDistsToMargin,
                                               mConsensusPredictions, mConsensusDistsToMargin);
    accuracy(mMockData.getTagsMat(), mConsensusPredictions, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Motion modality (c): " << mean << " ± " << conf << endl;
    
    cout << "... color (hon) modality" << endl;
    prediction.computeGridPredictionsConsensus(dMockData, dPredictions, dDistsToMargin,
                                               dConsensusPredictions, dConsensusDistsToMargin);
    accuracy(dMockData.getTagsMat(), dConsensusPredictions, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Depth modality (c): " << mean << " ± " << conf << endl;
    
    cout << "... thermal (hiog) modality" << endl;
    prediction.computeGridPredictionsConsensus(tMockData, tPredictions, tDistsToMargin,
                                               tConsensusPredictions, tConsensusDistsToMargin);
    accuracy(tMockData.getTagsMat(), tConsensusPredictions, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Thermal modality (c): " << mean << " ± " << conf << endl;
    
    cout << "... color (hog) modality" << endl;
    prediction.computeGridPredictionsConsensus(cMockData, cPredictions, cDistsToMargin,
                                               cConsensusPredictions, cConsensusDistsToMargin);
    accuracy(cMockData.getTagsMat(), cConsensusPredictions, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Color modality (c): " << mean << " ± " << conf << endl;

    cout << "... ramanan (ramanan scores) modality" << endl;
    prediction.computeGridPredictionsConsensus(cMockData, rPredictions, rDistsToMargin,
                                               rConsensusPredictions, rConsensusDistsToMargin);
    accuracy(cMockData.getTagsMat(), rConsensusPredictions, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Ramanan modality (c): " << mean << " ± " << conf << endl;


    // Save the results for the later prediction map generation
    aux.setTo(mConsensusPredictions);
    aux.save("mGridConsensusPredictions.yml");
    aux.setTo(dConsensusPredictions);
    aux.save("dGridConsensusPredictions.yml");
    aux.setTo(tConsensusPredictions);
    aux.save("tGridConsensusPredictions.yml");
    aux.setTo(cConsensusPredictions);
    aux.save("cGridConsensusPredictions.yml");
    aux.setTo(rConsensusPredictions);
    aux.save("rGridConsensusPredictions.yml");
    
   
    cout << "Computing fusion predictions ... " << endl;
    
    vector<GridMat> predictions, loglikelihoods, distsToMargin; // put together all the data
    predictions     += mPredictions, dPredictions, tPredictions, cPredictions;
    loglikelihoods  += mLoglikelihoods, dLoglikelihoods, tLoglikelihoods, cLoglikelihoods;
    distsToMargin   += mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin;
    
    vector<cv::Mat> consensuedPredictions, consensuedDistsToMargin;
    consensuedPredictions += mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
    cConsensusPredictions;
    consensuedDistsToMargin += mConsensusDistsToMargin, dConsensusDistsToMargin, tConsensusDistsToMargin,
                            cConsensusDistsToMargin;

//    predictions     += rPredictions;
//    loglikelihoods  += rScores;
//    distsToMargin   += rDistsToMargin;
//    consensuedPredictions += rConsensusPredictions;
//    consensuedDistsToMargin += rConsensusDistsToMargin;
 
    // Simple fusion
    
    cout << "... naive approach" << endl;
    
    SimpleFusionPrediction<cv::EM> simpleFusion;
    
    // Approach 1: Indiviual modalities' grid consensus first, then fusion
    cv::Mat simpleFusionPredictions1;
    cv::Mat simpleFusionDistsToMargin1; // (not used)
    
    simpleFusion.compute(consensuedPredictions, consensuedDistsToMargin, simpleFusionPredictions1, simpleFusionDistsToMargin1);
    
    accuracy(cMockData.getTagsMat(), simpleFusionPredictions1, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Cells' pre-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(simpleFusionPredictions1);
    aux.save("simpleFusionPredictions1.yml");
    
    // Approach 2: Prediction-based fusion, and then grid consensus
    GridMat gSimpleFusionPredictions2; // Cells' non-consensued predictions
    GridMat gSimpleFusionDistsToMargin2;
    
    cv::Mat consensuedSimpleFusionPredictions2, consensuedSimpleFusionDistsToMargin2;
    
    simpleFusion.compute(predictions, distsToMargin,
                         gSimpleFusionPredictions2, gSimpleFusionDistsToMargin2);
    prediction.computeGridPredictionsConsensus(cMockData, gSimpleFusionPredictions2, gSimpleFusionDistsToMargin2, consensuedSimpleFusionPredictions2, consensuedSimpleFusionDistsToMargin2); // post-consensus
    
    accuracy(cMockData.getTagsMat(), consensuedSimpleFusionPredictions2, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Cells' post-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(consensuedSimpleFusionPredictions2);
    aux.save("simpleFusionPredictions2.yml");

    // Approach 3: Raw distance to margin-based fusion, and then grid consensus
    GridMat gSimpleFusionPredictions3; // Cells' distances to margin
    GridMat gSimpleFusionDistsToMargin3;
    
    cv::Mat consensuedSimpleFusionPredictions3, consensuedSimpleFusionDistsToMargin3;
    
    simpleFusion.compute(distsToMargin, gSimpleFusionPredictions3, gSimpleFusionDistsToMargin3);
    prediction.computeGridPredictionsConsensus(cMockData, gSimpleFusionPredictions3, gSimpleFusionDistsToMargin3, consensuedSimpleFusionPredictions3, consensuedSimpleFusionDistsToMargin3); // post-consensus
    
    accuracy(cMockData.getTagsMat(), consensuedSimpleFusionPredictions3, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Distance-based and cells' post-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(consensuedSimpleFusionPredictions3);
    aux.save("simpleFusionPredictions3.yml");
    
    
    // Boost
    
    cout << "... Boost approach" << endl;
    
    cv::Mat boostFusionPredictions1, boostFusionPredictions2;
    
    ClassifierFusionPrediction<cv::EM,CvBoost> boostFusion;
    
    boostFusion.setData(distsToMargin, consensuedPredictions);
    boostFusion.setResponses(cMockData.getTagsMat());
    
    boostFusion.setValidationParameters(kTest, seed);
    boostFusion.setModelSelectionParameters(kModelSelec, true);
    
    boostFusion.setBoostType(CvBoost::GENTLE);
    boostFusion.setNumOfWeaks(numOfWeaks);
    boostFusion.setWeightTrimRate(weightTrimRates);
    
    boostFusion.setModelSelection(false);
    boostFusion.compute(boostFusionPredictions1);
    accuracy(cMockData.getTagsMat(), boostFusionPredictions1, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Boost fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(boostFusionPredictions1);
    aux.save("boostFusionPredictions.yml");
    
//    boostFusion.setModelSelection(false);
//    boostFusion.setStackedPrediction(true);
//    boostFusion.compute(boostFusionPredictions2);
//    accuracy(cMockData.getTagsMat(), boostFusionPredictions2, partitions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "Boost fusion /w preds: " << mean << " ± " << conf << endl;

    
    // MLP
    
    cout << "... MLP approach" << endl;
    
    cv::Mat mlpFusionPredictions1, mlpFusionPredictions2,
            mlpFusionPredictions3, mlpFusionPredictions4;
    
    ClassifierFusionPrediction<cv::EM,CvANN_MLP> mlpFusion;
    
    mlpFusion.setData(distsToMargin, consensuedPredictions);
    mlpFusion.setResponses(cMockData.getTagsMat());
    
    mlpFusion.setValidationParameters(kTest, seed);
    mlpFusion.setModelSelectionParameters(kModelSelec, true);
    
    mlpFusion.setActivationFunctionType(CvANN_MLP::SIGMOID_SYM);
    mlpFusion.setHiddenLayerSizes(hiddenLayerSizes);
    
    mlpFusion.setModelSelection(false);
    mlpFusion.compute(mlpFusionPredictions1);
    accuracy(cMockData.getTagsMat(), mlpFusionPredictions1, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "MLP sigmoid fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(mlpFusionPredictions1);
    aux.save("mlpSigmoidFusionPredictions.yml");
    
//    mlpFusion.setModelSelection(false);
//    mlpFusion.setStackedPrediction(true);
//    mlpFusion.compute(mlpFusionPredictions2);
//    accuracy(cMockData.getTagsMat(), mlpFusionPredictions2, partitions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "MLP sigmoid fusion /w preds: " << mean << " ± " << conf << endl;
    
    mlpFusion.setActivationFunctionType(CvANN_MLP::GAUSSIAN);
    
    mlpFusion.setModelSelection(false);
    mlpFusion.compute(mlpFusionPredictions3);
    accuracy(cMockData.getTagsMat(), mlpFusionPredictions3, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "MLP gaussian fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(mlpFusionPredictions3);
    aux.save("mlpGaussianFusionPredictions.yml");
    
//    mlpFusion.setModelSelection(false);
//    mlpFusion.setStackedPrediction(true);
//    mlpFusion.compute(mlpFusionPredictions4);
//    accuracy(cMockData.getTagsMat(), mlpFusionPredictions4, partitions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "MLP gaussian fusion /w preds: " << mean << " ± " << conf << endl;
    

    // SVM

    cout << "... SVM approach" << endl;

    cv::Mat svmFusionPredictions1, svmFusionPredictions2,
            svmFusionPredictions3, svmFusionPredictions4;

    ClassifierFusionPrediction<cv::EM,CvSVM> svmFusion;

    svmFusion.setData(distsToMargin, consensuedPredictions);
    svmFusion.setResponses(cMockData.getTagsMat());
    
    svmFusion.setValidationParameters(kTest, seed);
    svmFusion.setModelSelectionParameters(kModelSelec, true);

    svmFusion.setCs(cs);

    // Linear-kernel SVM

    svmFusion.setKernelType(CvSVM::LINEAR);

    svmFusion.setModelSelection(false);
    svmFusion.compute(svmFusionPredictions1);
    accuracy(cMockData.getTagsMat(), svmFusionPredictions1, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "SVM linear fusion: " << mean << " ± " << conf << endl;

    aux.setTo(svmFusionPredictions1);
    aux.save("svmLinearFusionPredictions.yml");
    
//    svmFusion.setModelSelection(false);
//    svmFusion.setStackedPrediction(true);
//    svmFusion.compute(svmFusionPredictions2);
//    accuracy(cMockData.getTagsMat(), svmFusionPredictions2, partitions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "SVM linear fusion /w preds: " << mean << " ± " << conf << endl;
    
    // RBF-kernel SVM
    
    svmFusion.setKernelType(CvSVM::RBF);
    
    svmFusion.setGammas(gammas);
    
    svmFusion.setModelSelection(false);
    svmFusion.compute(svmFusionPredictions3);
    accuracy(cMockData.getTagsMat(), svmFusionPredictions3, partitions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "SVM rbf fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(svmFusionPredictions3);
    aux.save("svmRBFFusionPredictions.yml");
    
//    svmFusion.setModelSelection(false);
//    svmFusion.setStackedPrediction(true);
//    svmFusion.compute(svmFusionPredictions4);
//    accuracy(cMockData.getTagsMat(), svmFusionPredictions4, partitions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "SVM rbf fusion /w preds: " << mean << " ± " << conf << endl;
    

*/
    
    //
    // Map writing
    //
    

    GridMapWriter mapWriter;
    
    //GridMat m; // create an empty GridMat
    //m.setTo(mConsensusPredictions); // set all the cells to same cv::Mat
    
    //mapWriter.write<unsigned char>(mMockData, m, "Predictions/");
   
    GridMat g;
    
    g.load("mGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(mMockData, g, "Motion/Predictions/");
           
    g.load("dGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(dMockData, g, "Depth/Predictions/");
   /*
    g.load("tGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(tMockData, g, "Thermal/Predictions/");
                         
    g.load("cGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "Color/Predictions/");
                                
    g.load("simpleFusionPredictions1.yml");
    mapWriter.write<unsigned char>(cMockData, g, "Simple_1_fusion/Predictions/");
                                       
    g.load("simpleFusionPredictions2.yml");
    mapWriter.write<unsigned char>(cMockData, g, "Simple_2_fusion/Predictions/");
                                              
    g.load("simpleFusionPredictions3.yml");
    mapWriter.write<unsigned char>(cMockData, g, "Simple_3_fusion/Predictions/");
    
    g.load("boostFusionPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "Boost_fusion/Predictions/");
           
    g.load("mlpSigmoidFusionPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "MLP_sigmoid_fusion/Predictions/");
                  
    g.load("mlpGaussianFusionPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "MLP_gaussian_fusion/Predictions/");
                         
    g.load("svmLinearFusionPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "SVM_linear_fusion/Predictions/");
                                
    g.load("svmRBFFusionPredictions.yml");
    mapWriter.write<unsigned char>(cMockData, g, "SVM_rbf_fusion/Predictions/");
    */
    
    //
    // Overlap
    //
    
    Validation validate;
    
    //Individual..
    
    //Depth
    cv::Mat overlapIDs;
    for (int s = 0; s < sequences.size(); s++)
    {
        ModalityData depthData;
        cout << "Reading depth data in scene " << s << ".." << endl;
        reader.overlapreadScene("Depth", "Depth", sequences[s], ".png", depthData);
        validate.getOverlap(depthData, dontCareRange, overlapIDs);
    }
    
    return 0;
}