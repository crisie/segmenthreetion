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
#include <boost/timer.hpp>

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
    hiddenLayerSizes += 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100;
    //numOfWeaks += 10, 20, 50, 100, 200, 500, 1000;
    //weightTrimRates += 0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99;
    
    // Validation procedure
    int kTest = 10; // number of folds in the outer cross-validation
    int kModelSelec = 3;
    int seed = 74;
 
    // Overlap params
    vector<int> dontCareRange;
    dontCareRange += 1, 2, 3, 4, 5, 6, 7, 8;
    //dontCareRange += 1;
    
    
// =============================================================================
//  Execution
// =============================================================================
    
    ModalityReader reader(dataPath);
    reader.setMasksOffset(masksOffset);
    
//    //
//    // Create partitions
//    // -----------------
//    // Execute once to create the partition file within each scene directory)
//    //
//    
//    vector<vector<cv::Rect> > boxesInFrames;
//    for (int s = 0; s < sequences.size(); s++)
//    {
//        boxesInFrames.clear();
//        reader.getBoundingBoxesFromGroundtruthMasks("Color", sequences[s], boxesInFrames);
//        cv::Mat numrects (boxesInFrames.size(), 1, cv::DataType<int>::type);
//        for (int f = 0; f < boxesInFrames.size(); f++)
//      {
//          numrects.at<int>(f,0) = boxesInFrames[f].size();
//        }
//        
//        cv::Mat partition;
//        cvpartition(numrects, kTest, seed, partition);
//
//        cv::FileStorage fs (dataPath + sequences[s] + "Partition.yml", cv::FileStorage::WRITE);
//        fs << "partition" << partition;
//        fs.release();
//    }
    
    //
    // Background subtraction
    // ----------------------
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

    // Color description

    ModalityGridData cGridData;

//    ColorFeatureExtractor cFE(cParam);
//	for (int s = 0; s < reader.getNumOfScenes(); s++)
//	{
//        cGridData.clear();
//        cout << "Reading color frames in scene " << s << " ..." << endl;
//		reader.readSceneData(s, "Color", "jpg", hp, wp, cGridData);
//        cout << "Describing color..." << endl;
//		cFE.describe(cGridData);
//        cGridData.saveDescription(reader.getScenePath(s), "Color.yml");
//    }

    // Motion description
    
    ModalityGridData mGridData;

//    MotionFeatureExtractor mFE(mParam);
//    for (int s = 0; s < reader.getNumOfScenes(); s++)
//	{
//        mGridData.clear();
//        cout << "Computing motion (from read color) frames in scene " << s << ".." << endl;
//		reader.readSceneData(s, "Motion", "jpg", hp, wp, mGridData);
//        cout << "Describing motion..." << endl;
//        mFE.describe(mGridData);
//        mGridData.saveDescription(reader.getScenePath(s), "Motion.yml");
//	}
    
    // Thermal description
    
    ModalityGridData tGridData;
    
//    ThermalFeatureExtractor tFE(tParam);
//	for (int s = 0; s < reader.getNumOfScenes(); s++)
//	{
//        tGridData.clear();
//        cout << "Reading thermal frames in scene " << s << ".." << endl;
//		reader.readSceneData(s, "Thermal", "jpg", hp, wp, tGridData);
//        cout << "Describing thermal..." << endl;
//		tFE.describe(tGridData);
//        tGridData.saveDescription(reader.getScenePath(s), "Thermal.yml");
//	}
    
    // Depth description
    
    ModalityGridData dGridData;
    
    DepthFeatureExtractor dFE(dParam);
	for (int s = 0; s < reader.getNumOfScenes(); s++)
	{
        dGridData.clear();
        cout << "Reading depth frames in scene " << s << ".." << endl;
		reader.readSceneData(s, "Depth", "png", hp, wp, dGridData);
        cout << "Describing depth..." << endl;
		dFE.describe(dGridData);
        dGridData.saveDescription(reader.getScenePath(s), "Depth.yml");
	}
    
    cGridData.clear();
    mGridData.clear();
    tGridData.clear();
    dGridData.clear();
    
    
    //
    // Individual prediction
    //
    
    GridMat mPredictions, dPredictions, tPredictions, cPredictions;//, rPredictions;
    GridMat mLoglikelihoods, dLoglikelihoods, tLoglikelihoods, cLoglikelihoods;//, rScores;
    GridMat mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin;//, rDistsToMargin;
    
    cv::Mat accuracies;
    GridMat accuraciesGrid;
    
    GridMat aux; // auxiliary gridmat, used for several purposes
    
    cv::Mat mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
    cConsensusPredictions;//, rConsensusPredictions;
    cv::Mat mConsensusDistsToMargin, dConsensusDistsToMargin, tConsensusDistsToMargin,
    cConsensusDistsToMargin;//, rConsensusDistsToMargin;

    // Other modalitites
    
    ModalityGridData mGridMetadata, dGridMetadata, tGridMetadata, cGridMetadata;
    
    //reader.readAllScenesMetadata("Color", "jpg", hp, wp, cGridMetadata);
    //reader.readAllScenesMetadata("Motion", "jpg", hp, wp, mGridMetadata);
    //reader.readAllScenesMetadata("Depth", "png", hp, wp, dGridMetadata);
    //reader.readAllScenesMetadata("Thermal", "jpg", hp, wp, tGridMetadata);
    
    /*
    reader.loadDescription("Color.yml", cGridMetadata);
    reader.loadDescription("Motion.yml", mGridMetadata);
    reader.loadDescription("Depth.yml", dGridMetadata);
    reader.loadDescription("Thermal.yml", tGridMetadata);
    
    ModalityPrediction<cv::EM> prediction;
    float mean, conf;
    cv::Mat means, confs;

    prediction.setNumOfMixtures(nmixtures);
    prediction.setLoglikelihoodThresholds(likelicuts);

    prediction.setValidationParameters(kTest);
    prediction.setModelSelectionParameters(kModelSelec, true);

    // Motion
    prediction.setData(mGridMetadata);

//    prediction.setModelSelection(false);
//    prediction.predict(mPredictions, mLoglikelihoods, mDistsToMargin);
//    prediction.getAccuracy(mPredictions, accuraciesGrid);
//    computeConfidenceInterval(accuraciesGrid, means, confs);
//    cout << means << endl;
//    cout << confs << endl;
//
//    mPredictions.save("mPredictions.yml");
//    mLoglikelihoods.save("mLoglikelihoods.yml");
//    mDistsToMargin.save("mDistsToMargin.yml");
    
    mPredictions.load("mPredictions.yml");
    mDistsToMargin.load("mDistsToMargin.yml");
    prediction.setPredictions(mPredictions);
    prediction.setDistsToMargin(mDistsToMargin);
    
    prediction.computeGridConsensusPredictions(mConsensusPredictions, mConsensusDistsToMargin);
    prediction.getAccuracy(mConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Motion modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(mConsensusPredictions); // predictions map generation purposes
    aux.save("mGridConsensusPredictions.yml");
    aux.setTo(mConsensusDistsToMargin);
    aux.save("mConsesusDistsToMargin.yml");


    // Depth
    prediction.setData(dGridMetadata);

//    prediction.setModelSelection(false);
//    prediction.predict(dPredictions, dLoglikelihoods, dDistsToMargin);
//    prediction.getAccuracy(dPredictions, accuraciesGrid);
//    computeConfidenceInterval(accuraciesGrid, means, confs);
//    cout << means << endl;
//    cout << confs << endl;
//    
//    dPredictions.save("dPredictions.yml");
//    dLoglikelihoods.save("dLoglikelihoods.yml");
//    dDistsToMargin.save("dDistsToMargin.yml");
    
    dPredictions.load("dPredictions.yml");
    dDistsToMargin.load("dDistsToMargin.yml");
    prediction.setPredictions(dPredictions);
    prediction.setDistsToMargin(dDistsToMargin);
    
    prediction.computeGridConsensusPredictions(dConsensusPredictions, dConsensusDistsToMargin);
    prediction.getAccuracy(dConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Depth modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(dConsensusPredictions); // predictions map generation purposes
    aux.save("dGridConsensusPredictions.yml");
    aux.setTo(dConsensusDistsToMargin);
    aux.save("dConsesusDistsToMargin.yml");


     // Thermal
    
    prediction.setData(tGridMetadata);

//    prediction.setModelSelection(false);
//    prediction.predict(tPredictions, tLoglikelihoods, tDistsToMargin);
//    prediction.getAccuracy(tPredictions, accuraciesGrid);
//    computeConfidenceInterval(accuraciesGrid, means, confs);
//    cout << means << endl;
//    cout << confs << endl;
//    
//    tPredictions.save("tPredictions.yml");
//    tLoglikelihoods.save("tLoglikelihoods.yml");
//    tDistsToMargin.save("tDistsToMargin.yml");
    
    tPredictions.load("tPredictions.yml");
    tDistsToMargin.load("tDistsToMargin.yml");
    prediction.setPredictions(tPredictions);
    prediction.setDistsToMargin(tDistsToMargin);
    
    prediction.computeGridConsensusPredictions(tConsensusPredictions, tConsensusDistsToMargin);
    prediction.getAccuracy(tConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Thermal modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(tConsensusPredictions); // predictions map generation purposes
    aux.save("tGridConsensusPredictions.yml");
    aux.setTo(tConsensusDistsToMargin);
    aux.save("tConsesusDistsToMargin.yml");
    

    // Color
    prediction.setData(cGridMetadata);
    prediction.setDimensionalityReduction(colorVariance);

//    prediction.setModelSelection(false);
//    prediction.predict(cPredictions, cLoglikelihoods, cDistsToMargin);
//    prediction.getAccuracy(cPredictions, accuraciesGrid);
//    computeConfidenceInterval(accuraciesGrid, means, confs);
//    cout << means << endl;
//    cout << confs << endl;
//    
//    cPredictions.save("cPredictions.yml");
//    cLoglikelihoods.save("cLoglikelihoods.yml");
//    cDistsToMargin.save("cDistsToMargin.yml");
    
    cPredictions.load("cPredictions.yml");
    cDistsToMargin.load("cDistsToMargin.yml");
    prediction.setPredictions(cPredictions);
    prediction.setDistsToMargin(cDistsToMargin);
    
    prediction.computeGridConsensusPredictions(cConsensusPredictions, cConsensusDistsToMargin);
    prediction.getAccuracy(cConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Color modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(cConsensusPredictions); // predictions map generation purposes
    aux.save("cGridConsensusPredictions.yml");
    aux.setTo(cConsensusDistsToMargin);
    aux.save("cConsesusDistsToMargin.yml");
    

//    // Ramanan
//
//    ModalityGridData rGridData;
//    reader.readAllScenesData("Ramanan", "mat", hp, wp, rGridData);
//
//    ModalityPrediction<cv::Mat> rprediction;
//
//    rprediction.setData(rGridData);
//
//    rprediction.setPositiveClassificationRatios(ratios);
//    rprediction.setScoreThresholds(scoreThresholds);
//
//    rprediction.setValidationParameters(kTest);
//    rprediction.setModelSelection(false); // false load it from disk (see .h)
//    rprediction.setModelSelectionParameters(kModelSelec, true);
//
//    rprediction.predict(rPredictions, rScores, rDistsToMargin);
//    rprediction.getAccuracy(rPredictions, rAccuracies);
//    computeConfidenceInterval(rAccuracies, means, confs);
//    cout << means << endl;
//    cout << confs << endl;
//
//    rPredictions.save("rPredictions.yml");
//    rScores.save("rScores.yml");
//    rDistsToMargin.save("rDistsToMargin.yml");
//    
//    rGridData.clear();
//    
//    prediction.computeGridConsensusPredictions(rConsensusPredictions, rConsensusDistsToMargin);
//    prediction.getAccuracy(rConsensusPredictions, accuracies);
//    computeConfidenceInterval(accuracies, &mean, &conf);
//    cout << "Motion modality (c): " << mean << " ± " << conf << endl;
    
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

    cout << "Computing fusion predictions ... " << endl;
    
    vector<ModalityGridData> mgds;
    mgds += mGridMetadata, dGridMetadata, tGridMetadata, cGridMetadata;
    
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
    
    SimpleFusionPrediction simpleFusion;
    simpleFusion.setModalitiesData(mgds);
    
    // Approach 1: Indiviual modalities' grid consensus first, then fusion
    cv::Mat simpleFusionPredictions1, simpleFusionDistsToMargin1; // (not used)
    
    simpleFusion.predict(consensuedPredictions, consensuedDistsToMargin, simpleFusionPredictions1, simpleFusionDistsToMargin1);
    computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
    cout << "Cells' pre-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(simpleFusionPredictions1);
    aux.save("simpleFusionPredictions1.yml");
    
    // Approach 2: Prediction-based fusion, and then grid consensus
    cv::Mat simpleFusionPredictions2, simpleFusionDistsToMargin2;
    
    simpleFusion.predict(predictions, distsToMargin, simpleFusionPredictions2, simpleFusionDistsToMargin2);
    computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
    cout << "Cells' post-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(simpleFusionPredictions2);
    aux.save("simpleFusionPredictions2.yml");

    // Approach 3: Raw distance to margin-based fusion, and then grid consensus    
    cv::Mat simpleFusionPredictions3, simpleFusionDistsToMargin3;
    
    simpleFusion.predict(distsToMargin, simpleFusionPredictions3, simpleFusionDistsToMargin3);
    computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
    cout << "Distance-based and cells' post-consensued predictions: " << mean << " ± " << conf << endl;
    
    aux.setTo(simpleFusionPredictions3);
    aux.save("simpleFusionPredictions3.yml");
    
 
    // Boost
    
    cout << "... Boost approach" << endl;
    
    cv::Mat boostFusionPredictions;
    
    ClassifierFusionPrediction<cv::EM,CvBoost> boostFusion;
    
    boostFusion.setData(mgds, distsToMargin, consensuedPredictions);
    
    boostFusion.setValidationParameters(kTest);
    boostFusion.setModelSelectionParameters(kModelSelec, seed, true);
    
    boostFusion.setBoostType(CvBoost::GENTLE);
    boostFusion.setNumOfWeaks(numOfWeaks);
    boostFusion.setWeightTrimRate(weightTrimRates);
    
    boostFusion.setModelSelection(false);
    boostFusion.predict(boostFusionPredictions);
    
    computeConfidenceInterval(boostFusion.getAccuracies(), &mean, &conf);
    cout << "Boost fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(boostFusionPredictions);
    aux.save("boostFusionPredictions.yml");

 
    // MLP
    
    cout << "... MLP approach" << endl;
    
    cv::Mat mlpSigmoidFusionPredictions, mlpGaussianFusionPredictions;
    
    ClassifierFusionPrediction<cv::EM,CvANN_MLP> mlpFusion;
    
    mlpFusion.setData(mgds, distsToMargin, consensuedPredictions);
    
    mlpFusion.setValidationParameters(kTest);
    mlpFusion.setModelSelectionParameters(kModelSelec, seed, true);
    
    mlpFusion.setActivationFunctionType(CvANN_MLP::SIGMOID_SYM);
    mlpFusion.setHiddenLayerSizes(hiddenLayerSizes);
    
    mlpFusion.setModelSelection(false);
    mlpFusion.predict(mlpSigmoidFusionPredictions);
    computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
    cout << "MLP sigmoid fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(mlpSigmoidFusionPredictions);
    aux.save("mlpSigmoidFusionPredictions.yml");
 
    mlpFusion.setActivationFunctionType(CvANN_MLP::GAUSSIAN);
    
    mlpFusion.setModelSelection(false);
    mlpFusion.predict(mlpGaussianFusionPredictions);
    computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
    cout << "MLP gaussian fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(mlpGaussianFusionPredictions);
    aux.save("mlpGaussianFusionPredictions.yml");

    
    // SVM

    cout << "... SVM approach" << endl;

    cv::Mat svmLinearFusionPredictions, svmRBFFusionPredictions;

    ClassifierFusionPrediction<cv::EM,CvSVM> svmFusion;

    svmFusion.setData(mgds, distsToMargin, consensuedPredictions);
    
    svmFusion.setValidationParameters(kTest);
    svmFusion.setModelSelectionParameters(kModelSelec, seed, true);

    svmFusion.setCs(cs);

    // Linear-kernel SVM

    svmFusion.setKernelType(CvSVM::LINEAR);

    svmFusion.setModelSelection(false);
    svmFusion.predict(svmLinearFusionPredictions);
    computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
    cout << "SVM linear fusion: " << mean << " ± " << conf << endl;

    aux.setTo(svmLinearFusionPredictions);
    aux.save("svmLinearFusionPredictions.yml");
    
    // RBF-kernel SVM
    
    svmFusion.setKernelType(CvSVM::RBF);
    
    svmFusion.setGammas(gammas);
    
    svmFusion.setModelSelection(false);
    svmFusion.predict(svmRBFFusionPredictions);
    computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
    cout << "SVM rbf fusion: " << mean << " ± " << conf << endl;
    
    aux.setTo(svmRBFFusionPredictions);
    aux.save("svmRBFFusionPredictions.yml");
    
    */
    
    //
    // Map writing
    //
    

    GridMapWriter mapWriter;
    
    //GridMat m; // create an empty GridMat
    //m.setTo(mConsensusPredictions); // set all the cells to same cv::Mat
    

    //mapWriter.write<unsigned char>(mGridMetadata, m, "Predictions/");
    
 
    GridMat g;
   /*
    g.load("boostFusionPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "Boost_fusion/Thermal/Predictions/");
   
    g.load("simpleFusionPredictions1.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_1_fusion/Thermal/Predictions/");
    
    g.load("simpleFusionPredictions2.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_2_fusion/Thermal/Predictions/");
    
    g.load("simpleFusionPredictions3.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_3_fusion/Thermal/Predictions/");
    
   
    g.load("mGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(mGridMetadata, g, "Motion/Predictions/");
    
    g.load("dGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(dGridMetadata, g, "Depth/Predictions/");

    g.load("tGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "Thermal/Predictions/");
                         
    g.load("cGridConsensusPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "Color/Predictions/");
                                
    g.load("simpleFusionPredictions1.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_1_fusion/Predictions/");
                                       
    g.load("simpleFusionPredictions2.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_2_fusion/Predictions/");
                                              
    g.load("simpleFusionPredictions3.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_3_fusion/Predictions/");
    
    g.load("boostFusionPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "Boost_fusion/Predictions/");
    
    g.load("mlpSigmoidFusionPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "MLP_sigmoid_fusion/Predictions/");
     
    g.load("mlpGaussianFusionPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "MLP_gaussian_fusion/Predictions/");
     
    g.load("svmLinearFusionPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "SVM_linear_fusion/Predictions/");
                                
    g.load("svmRBFFusionPredictions.yml");
    mapWriter.write<unsigned char>(cGridMetadata, g, "SVM_rbf_fusion/Predictions/");
   
    g.load("mlpSigmoidFusionPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "MLP_sigmoid_fusion/Thermal/Predictions/");
    
    g.load("mlpGaussianFusionPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "MLP_gaussian_fusion/Thermal/Predictions/");
    
    g.load("svmLinearFusionPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "SVM_linear_fusion/Thermal/Predictions/");
    
    g.load("svmRBFFusionPredictions.yml");
    mapWriter.write<unsigned char>(tGridMetadata, g, "SVM_rbf_fusion/Thermal/Predictions/");
 */
    

    //
    // Overlap
    //
    
    Validation validate;
    validate.setDontCareRange(dontCareRange);
    
    //Individual..
    
    cv::Mat overlapIDs, partitionedMeanOverlap, partitions = reader.getAllScenesPartition();
    
    //Depth
    vector<cv::Mat> partitionedOverlapIDs;
    
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData depthData;
        cout << "Reading depth data in scene " << s << ".." << endl;
        reader.overlapreadScene("Depth", "Depth", sequences[s], ".png", depthData);
        validate.getOverlap(depthData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "dGridConsensusOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("Color", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cGridConsensusOverlap.yml");

    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
   
    
    //Motion
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData motionData;
        cout << "Reading motion data in scene " << s << ".." << endl;
        reader.overlapreadScene("Motion", "Color", sequences[s], ".png", motionData);
        validate.getOverlap(motionData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "mGridConsensusOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tGridConsensusOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Simple fusion 1 - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_1_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlap1.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Simple fusion 1 - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_1_fusion", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlap1.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
     
    //Simple fusion 2 - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_2_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlap2.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Simple fusion 3 - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_3_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlap3.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
   
    //Simple fusion 2 - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_2_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlap2.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Simple fusion 1 - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_1_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlap1.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //Simple fusion 3 - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Simple_3_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlap3.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    
    //Boost fusion - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("Boost_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cBoostFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    
    
    //Boost fusion - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("Boost_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tBoostFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    
    //SVM linear fusion - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("SVM_linear_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSvmLinearFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //SVM rbf fusion - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("SVM_rbf_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSvmRBFFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //MLP gaussian fusion - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("MLP_Gaussian_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cMlpGaussianFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //MLP sigmoid fusion - color
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData colorData;
        cout << "Reading color data in scene " << s << ".." << endl;
        reader.overlapreadScene("MLP_sigmoid_fusion", "Color", sequences[s], ".png", colorData);
        validate.getOverlap(colorData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cMlpSigmoidFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //SVM linear fusion - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("SVM_linear_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSvmLinearFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //SVM rbf fusion - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("SVM_rbf_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSvmRBFFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //MLP gaussian fusion - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("MLP_Gaussian_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tMlpGaussianFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();
    
    //MLP sigmoid fusion - thermal
    for (int s = 0; s < sequences.size(); s++)
    {
        boost::timer t;
        ModalityData thermalData;
        cout << "Reading thermal data in scene " << s << ".." << endl;
        reader.overlapreadScene("MLP_sigmoid_fusion/Thermal", "Thermal", sequences[s], ".png", thermalData);
        validate.getOverlap(thermalData, overlapIDs);
        cout << "Elapsed time: " << t.elapsed() << endl;
    }
    validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
    validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
    validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tMlpSigmoidFusionOverlap.yml");
    
    overlapIDs.release();
    partitionedMeanOverlap.release();
    partitionedOverlapIDs.clear();

    
    return 0;
}