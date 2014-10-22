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
#include <boost/algorithm/string.hpp>

#include <pcl/console/parse.h>

//#include <matio.h>

using namespace boost::assign;
using namespace std;

int main(int argc, char** argv)
{
// =============================================================================
//  Program arguments
// =============================================================================
//
//    -D  , computes the modalities descriptions of the specified scenes (0,1,2)
//
//    -I  , computes the individual predictions.
//    -It , computes the invidual predictions and perform the specified model
//      selections ("m", "d", "t", or "c") and their mirrored versions ("M",
//      "D", "T", or "C").
//
//    -f  , computes the simple fusion predictions.
//
//    -F  , computes the learning fusion predictions
//    -Ft , computes the learning fusion predictions and perform the specified
//      model selections ("ada", "mlpsig", "mlpgau", "svmlin", or "svmrbf") and
//      and their mirrored ("ADA", "MLPsig", "MLPgau", "SVMlin", or "SVMrbf").
//
    
// =============================================================================
//  Parametrization
// =============================================================================
    
    // Dataset handling, create a reader pointing the data streams
    
#ifdef _WIN32 // Visual Studio
	string dataPath = "../Sequences/";
#elif __APPLE__ // Xcode
	string dataPath = "../../Sequences/";
#endif
	
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
    
    vector<vector<int> > validBoundBoxes;
    
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
    nmixtures += 2, 4, 6, 8, 10, 12;//, 8, 10; // classification parameter (training step)
    tmp = cvx::linspace(-2, 2, 12);//, 32); // -1, -0.8, -0.6, ..., 0.8
    vector<float> likelicuts;//; (tmp.begin()+1, tmp.end()-1);
    likelicuts += -3, -2.5, -2, -1.5, -1.25, -1, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3;
    vector<float> epsilons;
    epsilons += 1e-2, 1e-3, 1e-4, 1e-5;
    
    // ramanan
    tmp = cvx::linspace(0, 1.0, 30.0);
    vector<float> ratios (tmp.begin()+1, tmp.end()-1);
    tmp = cvx::linspace(0, 1.0, 30.0);
    vector<float> scoreThresholds (tmp.begin()+1, tmp.end()-1);
    
    double colorVariance = 0.95; // variance to keep in PCA's dim reduction in ColorModality
    
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
    int kModelSelec = 3 ;
    int seed = 74;
 
    // Overlap params
    vector<int> dontCareRange;
    dontCareRange += 1, 2, 3, 4, 5, 6, 7, 8;
    //dontCareRange += 1;
    
    std::vector<std::string> sequencesPaths;
    if (pcl::console::find_argument(argc, argv, "-S") > 0)
    {
        std::string valStr;
        pcl::console::parse(argc, argv, "-S", valStr);
        
        std::vector<std::string> valStrL;
        boost::split(valStrL, valStr, boost::is_any_of(","));
        
        std::vector<std::string>::iterator it;
        for (it = valStrL.begin(); it != valStrL.end(); it++)
            sequencesPaths += dataPath + (*it);
    }
    
    std::vector<int> descriptions;
    if (pcl::console::find_argument(argc, argv, "-D") > 0)
    {
        std::string valStr;
        pcl::console::parse(argc, argv, "-D", valStr);
        
        std::vector<std::string> valStrL;
        boost::split(valStrL, valStr, boost::is_any_of(","));
        
        std::vector<std::string>::iterator it;
        for (it = valStrL.begin(); it != valStrL.end(); it++)
            descriptions += stoi(*it);
    }
    
    bool bIndividualPredictions = false;
    
    bool bColorTraining = false; // normal
    bool bMotionTraining = false;
    bool bDepthTraining = false;
    bool bThermalTraining = false;
    bool bColorMirrTraining = false; // mirrored
    bool bMotionMirrTraining = false;
    bool bDepthMirrTraining = false;
    bool bThermalMirrTraining = false;
    
    if (pcl::console::find_argument(argc, argv, "-I") > 0)
    {
        bIndividualPredictions = true;
    }
    
    if (pcl::console::find_argument(argc, argv, "-It") > 0)
    {
        bIndividualPredictions = true;
        
        std::string valStr;
        pcl::console::parse(argc, argv, "-It", valStr);
        
        std::vector<std::string> valStrL;
        boost::split(valStrL, valStr, boost::is_any_of(","));
        
        std::vector<std::string>::iterator it;
        for (it = valStrL.begin(); it != valStrL.end(); it++)
        {
            if (*it == "c") bColorTraining = true; // normal
            if (*it == "m") bMotionTraining = true;
            if (*it == "d") bDepthTraining = true;
            if (*it == "t") bThermalTraining = true;
            if (*it == "C") bColorMirrTraining = true; // mirrored
            if (*it == "D") bMotionMirrTraining = true;
            if (*it == "D") bDepthMirrTraining = true;
            if (*it == "T") bThermalMirrTraining = true;
        }
    }
    
    bool bSimpleFusion = (pcl::console::find_argument(argc, argv, "-f") > 0);
    
    bool bLearningFusion = false;
    
    bool bAdaboostTraining = false; // normal
    bool bMlpSigmoidTraining = false;
    bool bMlpGaussianTraining = false;
    bool bSvmLinearTraining = false;
    bool bSvmRBFTraining = false;
    bool bAdaboostMirrTraining = false; // mirroed
    bool bMlpSigmoidMirrTraining = false;
    bool bMlpGaussianMirrTraining = false;
    bool bSvmLinearMirrTraining = false;
    bool bSvmRBFMirrTraining = false;
    
    if (pcl::console::find_argument(argc, argv, "-F") > 0)
    {
        bLearningFusion = true;
    }
    
    if (pcl::console::find_argument(argc, argv, "-Ft") > 0)
    {
        bLearningFusion = true;
        
        std::string valStr;
        pcl::console::parse(argc, argv, "-Ft", valStr);
        
        std::vector<std::string> valStrL;
        boost::split(valStrL, valStr, boost::is_any_of(","));
        
        std::vector<std::string>::iterator it;
        for (it = valStrL.begin(); it != valStrL.end(); it++)
        {
            if (*it == "ada") bAdaboostTraining = true; // normal
            if (*it == "mlpsig") bMlpSigmoidTraining = true;
            if (*it == "mlpgau") bMlpGaussianTraining = true;
            if (*it == "svmlin") bSvmLinearTraining = true;
            if (*it == "svmrbf") bSvmRBFTraining = true;
            if (*it == "ADA") bAdaboostMirrTraining = true; // mirrored
            if (*it == "MLPsig") bMlpSigmoidMirrTraining = true;
            if (*it == "MLPgau") bMlpGaussianMirrTraining = true;
            if (*it == "SVMlin") bSvmLinearMirrTraining = true;
            if (*it == "SVMrbf") bSvmRBFMirrTraining = true;
        }
    }
    
    bool bMapGeneration = (pcl::console::find_argument(argc, argv, "-M") > 0);
    
    bool bComputeOverlaps = (pcl::console::find_argument(argc, argv, "-O") > 0);
    
    bool bComputePartitions = (pcl::console::find_argument(argc, argv, "-P") > 0);
    
    bool bSubtractBackground = (pcl::console::find_argument(argc, argv, "-B") > 0);

// =============================================================================
//  Execution
// =============================================================================
    
    ModalityReader reader;
    reader.setSequences(sequencesPaths);
    reader.setMasksOffset(masksOffset);
    
    if (bComputePartitions)
    {
        //
        // Create partitions
        // -----------------
        // Execute once to create the partition file within each scene directory)
        //
        
        vector<vector<cv::Rect> > boxesInFrames;
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boxesInFrames.clear();
            reader.getBoundingBoxesFromGroundtruthMasks("Color", sequencesPaths[s], boxesInFrames);
            cv::Mat numrects (boxesInFrames.size(), 1, cv::DataType<int>::type);
            for (int f = 0; f < boxesInFrames.size(); f++)
            {
              numrects.at<int>(f,0) = boxesInFrames[f].size();
            }
            
            cv::Mat partition;
            cvpartition(numrects, kTest, seed, partition);

            cv::FileStorage fs (sequencesPaths[s] + "Partition.yml", cv::FileStorage::WRITE);
            fs << "partition" << partition;
            fs.release();
        }
    }
    
    if (bSubtractBackground)
    {
        // Background subtraction
        // ----------------------
        //
        
        ModalityData dData, cData, tData;
        
        ModalityWriter writer(dataPath);

        // Depth
        
        reader.read("Depth", dData);
        
        DepthBackgroundSubtractor dBS(fParam);
        dBS.setMasksOffset(masksOffset);
        //dBS.getMasks(dData);
        dBS.getBoundingRects(dData);
        //dBS.adaptGroundTruthToReg(dData);
        dBS.getGroundTruthBoundingRects(dData);
        dBS.getRoiTags(dData, false);
        
        // Thermal: depth-to-thermal registration
        
        // <------
        
        reader.read("Thermal", tData);
        
        ThermalBackgroundSubtractor tBS;
        tBS.setMasksOffset(masksOffset);
        //tBS.getMasks(dData, tData);
        tBS.getBoundingRects(dData, tData, validBoundBoxes); //modifies both dData and tData bounding rects
        //tBS.adaptGroundTruthToReg(tData);
        tBS.getRoiTags(dData, tData, validBoundBoxes); //modifies both dData and tData roi tags

        writer.saveValidBoundingBoxes(dataPath + "validBBs.yml", validBoundBoxes);

        writer.write("Thermal", tData);
        writer.write("Depth", dData);

        // Color: copy of depth info
        
        reader.read("Color", cData);

        ColorBackgroundSubtractor cBS;
        cBS.setMasksOffset(masksOffset);
        cBS.getMasks(dData, cData);
        cBS.getBoundingRects(dData, cData);
        //cBS.adaptGroundTruthToReg(dData, cData);
        cBS.getGroundTruthBoundingRects(dData,cData);
        cBS.getRoiTags(dData, cData);
        
        writer.write("Color", cData);
    }
    
    //
    // Feature extraction
    //
    
    cout << "Feature extraction ... " << endl;

    // Color description

    ModalityGridData cGridData;

    ColorFeatureExtractor cFE(cParam);
	for (int s = 0; s < descriptions.size(); s++)
	{
        cGridData.clear();
        cout << "Reading color frames in scene " << s << " ..." << endl;
		reader.readSceneData(sequencesPaths[descriptions[s]], "Color", "jpg", hp, wp, cGridData);
        cout << "Describing color..." << endl;
		cFE.describe(cGridData);
        cGridData.saveDescription(sequencesPaths[s], "Color.yml");
    }
    cGridData.clear();

    // Motion description
    
    ModalityGridData mGridData;

    MotionFeatureExtractor mFE(mParam);
	for (int s = 0; s < descriptions.size(); s++)
	{
        mGridData.clear();
        cout << "Computing motion (from read color) frames in scene " << s << ".." << endl;
		reader.readSceneData(sequencesPaths[descriptions[s]], "Motion", "jpg", hp, wp, mGridData);
        cout << "Describing motion..." << endl;
        mFE.describe(mGridData);
        mGridData.saveDescription(sequencesPaths[s], "Motion.yml");
	}
    mGridData.clear();

    // Thermal description
    
    ModalityGridData tGridData;

    ThermalFeatureExtractor tFE(tParam);
	for (int s = 0; s < descriptions.size(); s++)
	{
        tGridData.clear();
        cout << "Reading thermal frames in scene " << s << ".." << endl;
		reader.readSceneData(sequencesPaths[descriptions[s]], "Thermal", "jpg", hp, wp, tGridData);
        cout << "Describing thermal..." << endl;
		tFE.describe(tGridData);
        tGridData.saveDescription(sequencesPaths[s], "Thermal.yml");
	}
    tGridData.clear();
    
    // Depth description
    
    ModalityGridData dGridData;

    DepthFeatureExtractor dFE(dParam);
	for (int s = 0; s < descriptions.size(); s++)
	{
        dGridData.clear();
        cout << "Reading depth frames in scene " << s << ".." << endl;
		reader.readSceneData(sequencesPaths[descriptions[s]], "Depth", "png", hp, wp, dGridData);
        cout << "Describing depth..." << endl;
		dFE.describe(dGridData);
        dGridData.saveDescription(sequencesPaths[s], "Depth.yml");
	}
    dGridData.clear();


    //
    // Cells predictions
    //
    
    ///////////////
    GridMat aux; // several purposes
    ///////////////
    
    cout << "Prediction of individual cells ... " << endl;
    
    ModalityPrediction<cv::EM40> prediction;
    float mean, conf;
    cv::Mat means, confs;
    
    cv::Mat accuracies;
    GridMat accuraciesGrid;

    prediction.setNumOfMixtures(nmixtures);
    prediction.setEpsilons(epsilons);
    prediction.setLoglikelihoodThresholds(likelicuts);

    prediction.setValidationParameters(kTest);
    prediction.setModelSelectionParameters(kModelSelec, true);

    // Motion
    ModalityGridData mGridMetadata;
    
    reader.readAllScenesMetadata("Motion", "jpg", hp, wp, mGridMetadata);
    reader.loadDescription("Motion.yml", mGridMetadata);
    
    GridMat mPredictions, mPredictionsMirrored;
    GridMat mLoglikelihoods, mLoglikelihoodsMirrored;
    GridMat mDistsToMargin, mDistsToMarginMirrored;
    
    if (bIndividualPredictions)
    {
        cout << "Motion (individual prediction)" << endl;

        prediction.setData(mGridMetadata);
        
        prediction.setModelSelection(bMotionTraining);
        prediction.setTrainMirrored(false);

        prediction.predict(mPredictions, mLoglikelihoods, mDistsToMargin);
        prediction.getAccuracy(mPredictions, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;

        mPredictions.save("mPredictions.yml");
        mLoglikelihoods.save("mLoglikelihoods.yml");
        mDistsToMargin.save("mDistsToMargin.yml");
        
        prediction.setModelSelection(bMotionMirrTraining);
        prediction.setTrainMirrored(true);
        
        prediction.predict(mPredictionsMirrored, mLoglikelihoodsMirrored, mDistsToMarginMirrored);
        prediction.getAccuracy(mPredictionsMirrored, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        mPredictionsMirrored.save("mPredictionsMirrored.yml");
        mLoglikelihoodsMirrored.save("mLoglikelihoodsMirrored.yml");
        mDistsToMarginMirrored.save("mDistsToMarginMirrored.yml");
    }

    // Depth
    ModalityGridData dGridMetadata;
    
    reader.readAllScenesMetadata("Depth", "png", hp, wp, dGridMetadata);
    reader.loadDescription("Depth.yml", dGridMetadata);
    
    GridMat dPredictions, dPredictionsMirrored;
    GridMat dLoglikelihoods, dLoglikelihoodsMirrored;
    GridMat dDistsToMargin, dDistsToMarginMirrored;
    
    if (bIndividualPredictions)
    {
        cout << "Depth (individual prediction)" << endl;
        
        prediction.setData(dGridMetadata);
        
        prediction.setModelSelection(bDepthTraining);
        prediction.setTrainMirrored(false);
        
        prediction.predict(dPredictions, dLoglikelihoods, dDistsToMargin);
        prediction.getAccuracy(dPredictions, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        dPredictions.save("dPredictions.yml");
        dLoglikelihoods.save("dLoglikelihoods.yml");
        dDistsToMargin.save("dDistsToMargin.yml");
    
        prediction.setModelSelection(bDepthMirrTraining);
        prediction.setTrainMirrored(true);
        
        prediction.predict(dPredictionsMirrored, dLoglikelihoodsMirrored, dDistsToMarginMirrored);
        prediction.getAccuracy(dPredictionsMirrored, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        dPredictionsMirrored.save("dPredictionsMirrored.yml");
        dLoglikelihoodsMirrored.save("dLoglikelihoodsMirrored.yml");
        dDistsToMarginMirrored.save("dDistsToMarginMirrored.yml");
    }
    
    // Thermal
    ModalityGridData tGridMetadata;
    
    reader.readAllScenesMetadata("Thermal", "jpg", hp, wp,tGridMetadata);
    reader.loadDescription("Thermal.yml", tGridMetadata);
    
    GridMat tPredictions, tPredictionsMirrored;
    GridMat tLoglikelihoods, tLoglikelihoodsMirrored;
    GridMat tDistsToMargin, tDistsToMarginMirrored;

    if (bIndividualPredictions)
    {
        cout << "Thermal (individual prediction)" << endl;
        
        prediction.setData(tGridMetadata);
        
        prediction.setModelSelection(bThermalTraining);
        prediction.setTrainMirrored(false);
        
        prediction.predict(tPredictions, tLoglikelihoods, tDistsToMargin);
        prediction.getAccuracy(tPredictions, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        tPredictions.save("tPredictions.yml");
        tLoglikelihoods.save("tLoglikelihoods.yml");
        tDistsToMargin.save("tDistsToMargin.yml");

        prediction.setModelSelection(bThermalMirrTraining);
        prediction.setTrainMirrored(true);
        
        prediction.predict(tPredictionsMirrored, tLoglikelihoodsMirrored, tDistsToMarginMirrored);
        prediction.getAccuracy(tPredictionsMirrored, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        tPredictionsMirrored.save("tPredictionsMirrored.yml");
        tLoglikelihoodsMirrored.save("tLoglikelihoodsMirrored.yml");
        tDistsToMarginMirrored.save("tDistsToMarginMirrored.yml");
    }

    // Color
    ModalityGridData cGridMetadata;
    
    reader.readAllScenesMetadata("Color", "jpg", hp, wp, cGridMetadata);
    reader.loadDescription("Color.yml", cGridMetadata);
    
    GridMat cPredictions, cPredictionsMirrored;
    GridMat cLoglikelihoods, cLoglikelihoodsMirrored;
    GridMat cDistsToMargin, cDistsToMarginMirrored;
    
    if (bIndividualPredictions)
    {
        cout << "Color (individual prediction)" << endl;
        
        prediction.setData(cGridMetadata);
        prediction.setDimensionalityReduction(colorVariance);
        
        prediction.setModelSelection(bColorTraining);
        prediction.setTrainMirrored(false);
        
        prediction.predict(cPredictions, cLoglikelihoods, cDistsToMargin);
        prediction.getAccuracy(cPredictions, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        cPredictions.save("cPredictions.yml");
        cLoglikelihoods.save("cLoglikelihoods.yml");
        cDistsToMargin.save("cDistsToMargin.yml");

        prediction.setModelSelection(bColorMirrTraining);
        prediction.setTrainMirrored(true);
        
        prediction.predict(cPredictionsMirrored, cLoglikelihoodsMirrored, cDistsToMarginMirrored);
        prediction.getAccuracy(cPredictionsMirrored, accuraciesGrid);
        computeConfidenceInterval(accuraciesGrid, means, confs);
        cout << means << endl;
        cout << confs << endl;
        
        cPredictionsMirrored.save("cPredictionsMirrored.yml");
        cLoglikelihoodsMirrored.save("cLoglikelihoodsMirrored.yml");
        cDistsToMarginMirrored.save("cDistsToMarginMirrored.yml");
    }
    
    //
    // Grid cells consensus
    //
    
    cv::Mat mConsensusPredictions, dConsensusPredictions, tConsensusPredictions, cConsensusPredictions;
    cv::Mat mConsensusDistsToMargin, dConsensusDistsToMargin, tConsensusDistsToMargin, cConsensusDistsToMargin;
    
    cv::Mat mConsensusPredictionsMirrored, dConsensusPredictionsMirrored, tConsensusPredictionsMirrored, cConsensusPredictionsMirrored;
    cv::Mat mConsensusDistsToMarginMirrored, dConsensusDistsToMarginMirrored, tConsensusDistsToMarginMirrored, cConsensusDistsToMarginMirrored;
    
    cout << "Consensus of the grid cells ... " << endl;

    mPredictions.load("mPredictions.yml");
    mDistsToMargin.load("mDistsToMargin.yml");
    mPredictionsMirrored.load("mPredictionsMirrored.yml");
    mDistsToMarginMirrored.load("mDistsToMarginMirrored.yml");
    dPredictions.load("dPredictions.yml");
    dDistsToMargin.load("dDistsToMargin.yml");
    dPredictionsMirrored.load("dPredictionsMirrored.yml");
    dDistsToMarginMirrored.load("dDistsToMarginMirrored.yml");
    tPredictions.load("tPredictions.yml");
    tDistsToMargin.load("tDistsToMargin.yml");
    tPredictionsMirrored.load("tPredictionsMirrored.yml");
    tDistsToMarginMirrored.load("tDistsToMarginMirrored.yml");
    cPredictions.load("cPredictions.yml");
    cDistsToMargin.load("cDistsToMargin.yml");
    cPredictionsMirrored.load("cPredictionsMirrored.yml");
    cDistsToMarginMirrored.load("cDistsToMarginMirrored.yml");

    // Motion
    
    prediction.setData(mGridMetadata);
    
    prediction.setPredictions(mPredictions);
    prediction.setDistsToMargin(mDistsToMargin);
    prediction.computeGridConsensusPredictions(mConsensusPredictions, mConsensusDistsToMargin);
    prediction.getAccuracy(mConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Motion modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(mConsensusPredictions); // predictions map generation purposes
    aux.save("mGridConsensusPredictions.yml");
    aux.setTo(mConsensusDistsToMargin);
    aux.save("mGridConsensusDistsToMargin.yml");
    
    prediction.setPredictions(mPredictionsMirrored);
    prediction.setDistsToMargin(mDistsToMarginMirrored);
    prediction.computeGridConsensusPredictions(mConsensusPredictionsMirrored, mConsensusDistsToMarginMirrored);
    prediction.getAccuracy(mConsensusPredictionsMirrored, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Motion mirrored modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(mConsensusPredictionsMirrored); // predictions map generation purposes
    aux.save("mGridConsensusPredictionsMirrored.yml");
    aux.setTo(mConsensusDistsToMarginMirrored);
    aux.save("mGridConsensusDistsToMarginMirrored.yml");

    // Depth
    
    prediction.setData(dGridMetadata);
    
    prediction.setPredictions(dPredictions);
    prediction.setDistsToMargin(dDistsToMargin);
    prediction.computeGridConsensusPredictions(dConsensusPredictions, dConsensusDistsToMargin);
    prediction.getAccuracy(dConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Depth modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(dConsensusPredictions); // predictions map generation purposes
    aux.save("dGridConsensusPredictions.yml");
    aux.setTo(dConsensusDistsToMargin);
    aux.save("dGridConsensusDistsToMargin.yml");

    prediction.setPredictions(dPredictionsMirrored);
    prediction.setDistsToMargin(dDistsToMarginMirrored);
    prediction.computeGridConsensusPredictions(dConsensusPredictionsMirrored, dConsensusDistsToMarginMirrored);
    prediction.getAccuracy(dConsensusPredictionsMirrored, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Depth mirrored modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(dConsensusPredictionsMirrored); // predictions map generation purposes
    aux.save("dGridConsensusPredictionsMirrored.yml");
    aux.setTo(dConsensusDistsToMarginMirrored);
    aux.save("dGridConsensusDistsToMarginMirrored.yml");

    // Thermal
    
    prediction.setData(tGridMetadata);
    
    prediction.setPredictions(tPredictions);
    prediction.setDistsToMargin(tDistsToMargin);
    prediction.computeGridConsensusPredictions(tConsensusPredictions, tConsensusDistsToMargin);
    prediction.getAccuracy(tConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Thermal modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(tConsensusPredictions); // predictions map generation purposes
    aux.save("tGridConsensusPredictions.yml");
    aux.setTo(tConsensusDistsToMargin);
    aux.save("tGridConsensusDistsToMargin.yml");
    
    prediction.setPredictions(tPredictionsMirrored);
    prediction.setDistsToMargin(tDistsToMarginMirrored);
    prediction.computeGridConsensusPredictions(tConsensusPredictionsMirrored, tConsensusDistsToMarginMirrored);
    prediction.getAccuracy(tConsensusPredictionsMirrored, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Thermal mirrored modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(tConsensusPredictionsMirrored); // predictions map generation purposes
    aux.save("tGridConsensusPredictionsMirrored.yml");
    aux.setTo(tConsensusDistsToMarginMirrored);
    aux.save("tGridConsensusDistsToMarginMirrored.yml");

    // Color
    
    prediction.setData(cGridMetadata);
    
    prediction.setPredictions(cPredictions);
    prediction.setDistsToMargin(cDistsToMargin);
    prediction.computeGridConsensusPredictions(cConsensusPredictions, cConsensusDistsToMargin);
    prediction.getAccuracy(cConsensusPredictions, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Color modality (c): " << mean << " ± " << conf << endl;
    
    aux.setTo(cConsensusPredictions); // predictions map generation purposes
    aux.save("cGridConsensusPredictions.yml");
    aux.setTo(cConsensusDistsToMargin);
    aux.save("cGridConsensusDistsToMargin.yml");
    
    prediction.setPredictions(cPredictionsMirrored);
    prediction.setDistsToMargin(cDistsToMarginMirrored);
    prediction.computeGridConsensusPredictions(cConsensusPredictionsMirrored, cConsensusDistsToMarginMirrored);
    prediction.getAccuracy(cConsensusPredictionsMirrored, accuracies);
    computeConfidenceInterval(accuracies, &mean, &conf);
    cout << "Color mirrored modality (c): " << mean << " ± " << conf << endl;

    aux.setTo(cConsensusPredictionsMirrored); // predictions map generation purposes
    aux.save("cGridConsensusPredictionsMirrored.yml");
    aux.setTo(cConsensusDistsToMarginMirrored);
    aux.save("cGridConsensusDistsToMarginMirrored.yml");
    
    //
    // Fusion
    //
    
    cout << "Fusion of modalities ... " << endl;

    cout << "Computing fusion predictions ... " << endl;
    
    vector<ModalityGridData> mgds;
    mgds += mGridMetadata, dGridMetadata, tGridMetadata, cGridMetadata;
    
    vector<GridMat> predictions, distsToMargin; // put together all data
    predictions += mPredictions, dPredictions, tPredictions, cPredictions;
    distsToMargin += mDistsToMargin, dDistsToMargin, tDistsToMargin, cDistsToMargin;
    
    vector<GridMat> predictionsMirrored, distsToMarginMirrored; // put together all mirrored data
    predictionsMirrored += mPredictionsMirrored, dPredictionsMirrored, tPredictionsMirrored, cPredictionsMirrored;
    distsToMarginMirrored += mDistsToMarginMirrored, dDistsToMarginMirrored, tDistsToMarginMirrored, cDistsToMarginMirrored;
    
    vector<cv::Mat> consensuedPredictions, consensuedDistsToMargin;
    consensuedPredictions += mConsensusPredictions, dConsensusPredictions, tConsensusPredictions,
                             cConsensusPredictions;
    consensuedDistsToMargin += mConsensusDistsToMargin, dConsensusDistsToMargin, tConsensusDistsToMargin,
                               cConsensusDistsToMargin;
    
    vector<cv::Mat> consensuedPredictionsMirrored, consensuedDistsToMarginMirrored;
    consensuedPredictionsMirrored += mConsensusPredictionsMirrored, dConsensusPredictionsMirrored, tConsensusPredictionsMirrored, cConsensusPredictionsMirrored;
    consensuedDistsToMarginMirrored += mConsensusDistsToMarginMirrored, dConsensusDistsToMarginMirrored, tConsensusDistsToMarginMirrored, cConsensusDistsToMarginMirrored;
 
    // Simple fusion
    
    if (bSimpleFusion)
    {
        cout << "... naive approach" << endl;
        
        SimpleFusionPrediction simpleFusion;
        simpleFusion.setModalitiesData(mgds);
        
        // Approach 1: Indiviual modalities' grid consensus first, then fusion
        cv::Mat simpleFusionPredictions1, simpleFusionDistsToMargin1;
        cv::Mat simpleFusionPredictionsMirrored1, simpleFusionDistsToMarginMirrored1;
        
        simpleFusion.predict(consensuedPredictions, consensuedDistsToMargin, simpleFusionPredictions1, simpleFusionDistsToMargin1);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Cells' pre-consensued predictions: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictions1);
        aux.save("simpleFusionPredictions1.yml");
        
        simpleFusion.predict(consensuedPredictionsMirrored, consensuedDistsToMarginMirrored, simpleFusionPredictionsMirrored1, simpleFusionDistsToMarginMirrored1);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Cells' pre-consensued predictions mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictionsMirrored1);
        aux.save("simpleFusionPredictionsMirrored1.yml");
        
        // Approach 2: Prediction-based fusion, and then grid consensus
        cv::Mat simpleFusionPredictions2, simpleFusionDistsToMargin2;
        cv::Mat simpleFusionPredictionsMirrored2, simpleFusionDistsToMarginMirrored2;
        
        simpleFusion.predict(predictions, distsToMargin, simpleFusionPredictions2, simpleFusionDistsToMargin2);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Cells' post-consensued predictions: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictions2);
        aux.save("simpleFusionPredictions2.yml");
        
        simpleFusion.predict(predictionsMirrored, distsToMarginMirrored, simpleFusionPredictionsMirrored2, simpleFusionDistsToMarginMirrored2);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Cells' post-consensued predictions mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictionsMirrored2);
        aux.save("simpleFusionPredictionsMirrored2.yml");

        // Approach 3: Raw distance to margin-based fusion, and then grid consensus    
        cv::Mat simpleFusionPredictions3, simpleFusionDistsToMargin3;
        cv::Mat simpleFusionPredictionsMirrored3, simpleFusionDistsToMarginMirrored3;
        
        simpleFusion.predict(distsToMargin, simpleFusionPredictions3, simpleFusionDistsToMargin3);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Distance-based and cells' post-consensued predictions: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictions3);
        aux.save("simpleFusionPredictions3.yml");
        
        simpleFusion.predict(distsToMarginMirrored, simpleFusionPredictionsMirrored3, simpleFusionDistsToMarginMirrored3);
        computeConfidenceInterval(simpleFusion.getAccuracies(), &mean, &conf);
        cout << "Distance-based and cells' post-consensued predictions mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(simpleFusionPredictionsMirrored3);
        aux.save("simpleFusionPredictionsMirrored3.yml");
    }

    if (bLearningFusion)
    {
        // Boost
        cout << "... Boost approach" << endl;
        
        cv::Mat boostFusionPredictions, boostFusionPredictionsMirrored;
        
        ClassifierFusionPrediction<cv::EM40,CvBoost> boostFusion;
        boostFusion.setValidationParameters(kTest);
        boostFusion.setModelSelectionParameters(kModelSelec, seed, true);
        
        boostFusion.setBoostType(CvBoost::GENTLE);
        boostFusion.setNumOfWeaks(numOfWeaks);
        boostFusion.setWeightTrimRate(weightTrimRates);
        
        boostFusion.setData(mgds, distsToMargin, consensuedPredictions);
        boostFusion.setModelSelection(bAdaboostTraining);
        boostFusion.setTrainMirrored(false);
        
        boostFusion.predict(boostFusionPredictions);
        computeConfidenceInterval(boostFusion.getAccuracies(), &mean, &conf);
        cout << "Boost fusion: " << mean << " ± " << conf << endl;
        
        aux.setTo(boostFusionPredictions);
        aux.save("boostFusionPredictions.yml");
        
        boostFusion.setData(mgds, distsToMarginMirrored, consensuedPredictionsMirrored);
        boostFusion.setModelSelection(bAdaboostMirrTraining);
        boostFusion.setTrainMirrored(true);

        boostFusion.predict(boostFusionPredictionsMirrored);
        computeConfidenceInterval(boostFusion.getAccuracies(), &mean, &conf);
        cout << "Boost fusion mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(boostFusionPredictionsMirrored);
        aux.save("boostFusionPredictionsMirrored.yml");


        // MLP
        
        cout << "... MLP approach" << endl;
        
        cv::Mat mlpSigmoidFusionPredictions, mlpGaussianFusionPredictions,
                mlpSigmoidFusionPredictionsMirrored, mlpGaussianFusionPredictionsMirrored;
        
        ClassifierFusionPrediction<cv::EM40,CvANN_MLP> mlpFusion;
        mlpFusion.setValidationParameters(kTest);
        mlpFusion.setModelSelectionParameters(kModelSelec, seed, true);
        
        mlpFusion.setHiddenLayerSizes(hiddenLayerSizes);
        
        // sigmoid
        mlpFusion.setActivationFunctionType(CvANN_MLP::SIGMOID_SYM);
        
        mlpFusion.setData(mgds, distsToMargin, consensuedPredictions);
        mlpFusion.setModelSelection(bMlpSigmoidTraining);
        mlpFusion.setTrainMirrored(false);

        mlpFusion.predict(mlpSigmoidFusionPredictions);
        computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
        cout << "MLP sigmoid fusion: " << mean << " ± " << conf << endl;
        
        aux.setTo(mlpSigmoidFusionPredictions);
        aux.save("mlpSigmoidFusionPredictions.yml");
        
        mlpFusion.setData(mgds, distsToMarginMirrored, consensuedPredictionsMirrored);
        mlpFusion.setModelSelection(bMlpSigmoidMirrTraining);
        mlpFusion.setTrainMirrored(true);
        
        mlpFusion.predict(mlpSigmoidFusionPredictionsMirrored);
        computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
        cout << "MLP sigmoid fusion mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(mlpSigmoidFusionPredictionsMirrored);
        aux.save("mlpSigmoidFusionPredictionsMirrored.yml");
     
        // gaussian
        mlpFusion.setActivationFunctionType(CvANN_MLP::GAUSSIAN);

        mlpFusion.setData(mgds, distsToMargin, consensuedPredictions);
        mlpFusion.setModelSelection(bMlpGaussianTraining);
        mlpFusion.setTrainMirrored(false);
        
        mlpFusion.predict(mlpGaussianFusionPredictions);
        computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
        cout << "MLP gaussian fusion: " << mean << " ± " << conf << endl;
        
        aux.setTo(mlpGaussianFusionPredictions);
        aux.save("mlpGaussianFusionPredictions.yml");
        
        mlpFusion.setData(mgds, distsToMarginMirrored, consensuedPredictionsMirrored);
        mlpFusion.setModelSelection(bMlpGaussianMirrTraining);
        mlpFusion.setTrainMirrored(true);
        
        mlpFusion.predict(mlpGaussianFusionPredictionsMirrored);
        computeConfidenceInterval(mlpFusion.getAccuracies(), &mean, &conf);
        cout << "MLP gaussian fusion mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(mlpGaussianFusionPredictionsMirrored);
        aux.save("mlpGaussianFusionPredictionsMirrored.yml");


        // SVM

        cout << "... SVM approach" << endl;

        cv::Mat svmLinearFusionPredictions, svmRBFFusionPredictions,
                svmLinearFusionPredictionsMirrored, svmRBFFusionPredictionsMirrored;

        ClassifierFusionPrediction<cv::EM40,CvSVM> svmFusion;
        svmFusion.setValidationParameters(kTest);
        svmFusion.setModelSelectionParameters(kModelSelec, seed, false);
        svmFusion.setCs(cs);
        
        // linear-kernel

        svmFusion.setKernelType(CvSVM::LINEAR);

        svmFusion.setData(mgds, distsToMargin, consensuedPredictions);
        svmFusion.setModelSelection(bSvmLinearTraining);
        svmFusion.setTrainMirrored(false);
        
        svmFusion.predict(svmLinearFusionPredictions);
        computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
        cout << "SVM linear fusion: " << mean << " ± " << conf << endl;
        
        aux.setTo(svmLinearFusionPredictions);
        aux.save("svmLinearFusionPredictions.yml");
        
        svmFusion.setData(mgds, distsToMarginMirrored, consensuedPredictionsMirrored);
        svmFusion.setModelSelection(bSvmLinearMirrTraining);
        svmFusion.setTrainMirrored(true);
        
        svmFusion.predict(svmLinearFusionPredictionsMirrored);
        computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
        cout << "SVM linear fusion mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(svmLinearFusionPredictionsMirrored);
        aux.save("svmLinearFusionPredictionsMirrored.yml");
        
        // RBF-kernel
        
        svmFusion.setKernelType(CvSVM::RBF);
        
        svmFusion.setGammas(gammas);
        
        svmFusion.setData(mgds, distsToMargin, consensuedPredictions);
        svmFusion.setModelSelection(bSvmRBFTraining);
        svmFusion.setTrainMirrored(false);
        
        svmFusion.predict(svmRBFFusionPredictions);
        computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
        cout << "SVM rbf fusion: " << mean << " ± " << conf << endl;
        
        aux.setTo(svmRBFFusionPredictions);
        aux.save("svmRBFFusionPredictions.yml");
        
        svmFusion.setData(mgds, distsToMarginMirrored, consensuedPredictionsMirrored);
        svmFusion.setModelSelection(bSvmRBFMirrTraining);
        svmFusion.setTrainMirrored(true);
        
        svmFusion.predict(svmRBFFusionPredictionsMirrored);
        computeConfidenceInterval(svmFusion.getAccuracies(), &mean, &conf);
        cout << "SVM rbf fusion mirrored: " << mean << " ± " << conf << endl;
        
        aux.setTo(svmRBFFusionPredictionsMirrored);
        aux.save("svmRBFFusionPredictionsMirrored.yml");
    }
    
    //
    // Map writing
    //
    
    if (bMapGeneration)
    {
        std::cout << "Generating maps of predictions... " << std::endl;
        
        GridMapWriter mapWriter;
        GridMat g;
       
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
        
        // Mirrored
        g.load("boostFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "Boost_fusion_mirrored/Thermal/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored1.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_1_fusion_mirrored/Thermal/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored2.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_2_fusion_mirrored/Thermal/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored3.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "Simple_3_fusion_mirrored/Thermal/Predictions/");
        
        g.load("mGridConsensusPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(mGridMetadata, g, "Motion_mirrored/Predictions/");
        
        g.load("dGridConsensusPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(dGridMetadata, g, "Depth_mirrored/Predictions/");
        
        g.load("tGridConsensusPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "Thermal_mirrored/Predictions/");
        
        g.load("cGridConsensusPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "Color_mirrored/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored1.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_1_fusion_mirrored/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored2.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_2_fusion_mirrored/Predictions/");
        
        g.load("simpleFusionPredictionsMirrored3.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "Simple_3_fusion_mirrored/Predictions/");
        
        g.load("boostFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "Boost_fusion_mirrored/Predictions/");
        
        g.load("mlpSigmoidFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "MLP_sigmoid_fusion_mirrored/Predictions/");
        
        g.load("mlpGaussianFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "MLP_gaussian_fusion_mirrored/Predictions/");
        
        g.load("svmLinearFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "SVM_linear_fusion_mirrored/Predictions/");
        
        g.load("svmRBFFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(cGridMetadata, g, "SVM_rbf_fusion_mirrored/Predictions/");
        
        g.load("mlpSigmoidFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "MLP_sigmoid_fusion_mirrored/Thermal/Predictions/");
        
        g.load("mlpGaussianFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "MLP_gaussian_fusion_mirrored/Thermal/Predictions/");
        
        g.load("svmLinearFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "SVM_linear_fusion_mirrored/Thermal/Predictions/");
        
        g.load("svmRBFFusionPredictionsMirrored.yml");
        mapWriter.write<unsigned char>(tGridMetadata, g, "SVM_rbf_fusion_mirrored/Thermal/Predictions/");
    }
    
    //
    // Overlap
    //
    
    if (bComputeOverlaps)
    {
        std::cout << "Computing ovelaps ... " << std::endl;
    
        Validation validate;
        validate.setDontCareRange(dontCareRange);
     
        //Individual..
        
        cv::Mat overlapIDs, partitionedMeanOverlap, partitions = reader.getAllScenesPartition();
        vector<cv::Mat> partitionedOverlapIDs;

//        //Depth
//        std::cout << "Depth" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData depthData;
//            cout << "Reading depth data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Depth_mirrored", "Depth", sequencesPaths[s], "png", depthData);
//            validate.getOverlap(depthData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "dGridConsensusOverlapMirrored.yml");
//    
//        
//        //Color
//        std::cout << "Color" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData colorData;
//            cout << "Reading color data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Color_mirrored", "Color", sequencesPaths[s], "png", colorData);
//            validate.getOverlap(colorData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cGridConsensusOverlapMirrored.yml");
//       
//        
//        //Motion
//        std::cout << "Motion" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData motionData;
//            cout << "Reading motion data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Motion_mirrored", "Color", sequencesPaths[s], "png", motionData);
//            validate.getOverlap(motionData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "mGridConsensusOverlapMirrored.yml");
//
//        
//        //Thermal
//        std::cout << "Thermal" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData thermalData;
//            cout << "Reading thermal data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Thermal_mirrored", "Thermal", sequencesPaths[s], "png", thermalData);
//            validate.getOverlap(thermalData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tGridConsensusOverlapMirrored.yml");
//        
//        
//        //Simple fusion 1 - color
//        std::cout << "Simple fusion 1 - color" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData colorData;
//            cout << "Reading color data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Simple_1_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
//            validate.getOverlap(colorData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlapMirrored1.yml");
//        
//        
//        //Simple fusion 1 - thermal
//        std::cout << "Simple fusion 1 - thermal" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData thermalData;
//            cout << "Reading thermal data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Simple_1_fusion_mirrored", "Thermal", sequencesPaths[s], "png", thermalData);
//            validate.getOverlap(thermalData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlapMirrored1.yml");
//        
//         
//        //Simple fusion 2 - color
//        std::cout << "Simple fusion 2 - color" << std::endl;
//        overlapIDs.release();
//        for (int s = 0; s < sequencesPaths.size(); s++)
//        {
//            boost::timer t;
//            ModalityData colorData;
//            cout << "Reading color data in scene " << s << ".." << endl;
//            reader.overlapreadScene("Simple_2_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
//            validate.getOverlap(colorData, overlapIDs);
//            cout << "Elapsed time: " << t.elapsed() << endl;
//        }
//        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
//        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
//        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlapMirrored2.yml");
        
        // ERROR
        //Simple fusion 3 - color
        std::cout << "Simple fusion 3 - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("Simple_3_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSimpleFusionOverlapMirrored3.yml");

        
        //Simple fusion 2 - thermal
        std::cout << "Simple fusion 2 - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("Simple_2_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlapMirrored2.yml");
        

        
        //Simple fusion 1 - thermal
        std::cout << "Simple fusion 1 - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("Simple_1_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlapMirrored1.yml");
        

        //Simple fusion 3 - thermal
        std::cout << "Simple fusion 3 - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("Simple_3_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSimpleFusionOverlapMirrored3.yml");
        

        //Boost fusion - color
        std::cout << "Boost fusion - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("Boost_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cBoostFusionOverlapMirrored.yml");
    
        
        //Boost fusion - thermal
        std::cout << "Boost fusion - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("Boost_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tBoostFusionOverlapMirrored.yml");
        

        //SVM linear fusion - color
        std::cout << "SVM linear fusion - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("SVM_linear_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSvmLinearFusionOverlapMirrored.yml");
        
        
        //SVM rbf fusion - color
        std::cout << "SVM rbf fusion - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("SVM_rbf_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cSvmRBFFusionOverlapMirrored.yml");
        

        //MLP gaussian fusion - color
        std::cout << "MLP gaussian fusion - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("MLP_Gaussian_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cMlpGaussianFusionOverlapMirrored.yml");
        
        
        //MLP sigmoid fusion - color
        std::cout << "MLP sigmoid fusion - color" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData colorData;
            cout << "Reading color data in scene " << s << ".." << endl;
            reader.overlapreadScene("MLP_sigmoid_fusion_mirrored", "Color", sequencesPaths[s], "png", colorData);
            validate.getOverlap(colorData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "cMlpSigmoidFusionOverlapMirrored.yml");
        

        //SVM linear fusion - thermal
        std::cout << "SVM linear fusion - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("SVM_linear_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSvmLinearFusionOverlapMirrored.yml");
        
        
        //SVM rbf fusion - thermal
        std::cout << "SVM rbf fusion - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("SVM_rbf_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tSvmRBFFusionOverlapMirrored.yml");
        

        //MLP gaussian fusion - thermal
        std::cout << "MLP gaussian fusion - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("MLP_Gaussian_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tMlpGaussianFusionOverlapMirrored.yml");

        
        //MLP sigmoid fusion - thermal
        std::cout << "MLP sigmoid fusion - thermal" << std::endl;
        overlapIDs.release();
        for (int s = 0; s < sequencesPaths.size(); s++)
        {
            boost::timer t;
            ModalityData thermalData;
            cout << "Reading thermal data in scene " << s << ".." << endl;
            reader.overlapreadScene("MLP_sigmoid_fusion_mirrored/Thermal", "Thermal", sequencesPaths[s], "png", thermalData);
            validate.getOverlap(thermalData, overlapIDs);
            cout << "Elapsed time: " << t.elapsed() << endl;
        }
        validate.createOverlapPartitions(partitions, overlapIDs, partitionedOverlapIDs);
        validate.getMeanOverlap(partitionedOverlapIDs, partitionedMeanOverlap);
        validate.save(partitionedOverlapIDs, partitionedMeanOverlap, "tMlpSigmoidFusionOverlapMirrored.yml");
        
    }
    
    return 0;
}