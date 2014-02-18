//
//  Segmentator.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 20/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "TrimodalSegmentator.h"
#include "TrimodalPixelClassifier.h"

#include "ColorFeatureExtractor.h"
#include "MotionFeatureExtractor.h"
#include "DepthFeatureExtractor.h"
#include "ThermalFeatureExtractor.h"

#include "GridMat.h"

#include "StatTools.h"
#include "DebugTools.h"

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::filesystem; 
using namespace std;

/*
 * Constructor
 */

TrimodalSegmentator::TrimodalSegmentator(const unsigned char offsetID)
    : m_OffsetID(offsetID)
{ }


void TrimodalSegmentator::extractFeatures(const unsigned int hp, const unsigned int wp,
                                          const ColorParametrization cParam, const MotionParametrization mParam,
                                          const DepthParametrization dParam, const ThermalParametrization tParam)
{
    m_hp = hp;
    m_wp = wp;
    
    // Feature extractors
    
    ColorFeatureExtractor   cFE(m_hp, m_wp, cParam);
    MotionFeatureExtractor  mFE(m_hp, m_wp, mParam);
    DepthFeatureExtractor   dFE(m_hp, m_wp, dParam);
    ThermalFeatureExtractor tFE(m_hp, m_wp, tParam);
    
    // Descriptions
    
    GridMat cSubDescriptors(m_hp, m_wp, GridMat::SUBJECT);
    GridMat cObjDescriptors(m_hp, m_wp, GridMat::OBJECT);
    GridMat cUnkDescriptors(m_hp, m_wp, GridMat::UNKNOWN);
    
    GridMat mSubDescriptors(m_hp, m_wp, GridMat::SUBJECT);
    GridMat mObjDescriptors(m_hp, m_wp, GridMat::OBJECT);
    GridMat mUnkDescriptors(m_hp, m_wp, GridMat::UNKNOWN);
    
    GridMat dSubDescriptors(m_hp, m_wp, GridMat::SUBJECT);
    GridMat dObjDescriptors(m_hp, m_wp, GridMat::OBJECT);
    GridMat dUnkDescriptors(m_hp, m_wp, GridMat::UNKNOWN);
    
    GridMat tSubDescriptors(m_hp, m_wp, GridMat::SUBJECT);
    GridMat tObjDescriptors(m_hp, m_wp, GridMat::OBJECT);
    GridMat tUnkDescriptors(m_hp, m_wp, GridMat::UNKNOWN);
    
    for (int i = 0; i < m_ScenesPaths.size(); i++)
    {
        //extractModalityFeatures(m_ScenesPaths[i], "Color", cParam, cDescriptors);
        //extractModalityFeatures(m_ScenesPaths[i], "Motion" mParam, mDescriptors);
        //extractModalityFeatures(m_ScenesPaths[i], "Depth", dParam, dDescriptors);
        extractModalityFeatures(m_ScenesPaths[i], "Thermal", &
                                tFE, tSubDescriptors, tObjDescriptors, tUnkDescriptors);
    }
}


void TrimodalSegmentator::extractModalityFeatures(string scenePath, string modality, FeatureExtractor* fe,
                                                  GridMat& subDescriptors, GridMat& objDescriptors, GridMat& unkDescriptors)
{
    // Load data from disk: frames, masks, and rectangular bounding boxes
    
	vector<cv::Mat> frames;
	vector<cv::Mat> masks;
    vector< vector<cv::Rect> > boundingRects;
    vector< vector<int> > tags;
    
    loadDataToMats   (scenePath + "Frames/" + modality + "/", "jpg", frames);
	loadDataToMats   (scenePath + "Masks/" + modality + "/", "png", masks);
	loadBoundingRects(scenePath + "Masks/" + modality + ".yml", boundingRects, tags);
    //visualizeMasksWithinRects(masks, bounding_rects); // DEBUG
    
    // Grid frames and masks
    
    vector<GridMat> gFramesTrain, gMasksTrain; // the ones used to train
    
	grid(frames, boundingRects, tags, m_hp, m_wp, gFramesTrain);
	grid(masks, boundingRects, tags, m_hp, m_wp, gMasksTrain);
    //visualizeGridmats(gframes_train); // DEBUG
    //visualizeGridmats(gmasks_train); // DEBUG
    
	//
	// Feature extraction
	//
    
    fe->describe(gFramesTrain, gMasksTrain, subDescriptors, objDescriptors, unkDescriptors); // framews dimensionality reduced to the description dimensionality

//	// DEBUG
//    boost::posix_time::ptime t = boost::posix_time::second_clock::local_time();
//    std::stringstream ss;
//    ss << t;
//    descriptors.saveFS(std::string("color_descriptors_") + ss.str() + std::string(".yml")); // DEBUG: save to a file
//	//
//    // Clusterize people
//    // -----------------
//    // People are separated by "pose", i.e. each grid partition contains almost
//    // the same information in all the images within a particular pose.
//    //
//    
//    GridMat labelsTrain, centers; // Assigned cluster labels and cluster centers
//  	
//    TrimodalClusterer trimodalClusterer(m_NumClusters);
//    trimodalClusterer.trainClusters(descriptionsTrain, labelsTrain, centers);
//
//    //
//    // Training step
//    // -------------
//    // Model the GMMs using the Expectation-Maximization.
//    // Thus, having (hp * wp) * n_P * p modeled GMMs. Where (hp * wp) is the parametrized size of the grid.
//    // And p is the number of poses.
//    //
//
//    TrimodalPixelClassifier trimodalPixelClassificator(m_NumMixtures);
//    trimodalPixelClassificator.train(descriptionsTrain, labelsTrain, centers);
//
//	cout << centers << endl;
//	
//	//
//	// Train LogLikelihood PDFs
//	//
//
//    GridMat classification, probabilities, logLikelihoods;
//    trimodalPixelClassificator.test2(descriptionsTrain, labelsTrain, classification, probabilities, logLikelihoods);
//    
//    cout << probabilities << endl;
//    
//    /*
//    cout << classification << endl;
//    cout << probabilities << endl;
//	cout << logLikelihoods << endl;
//    
//    cv::Mat means (m_hp, m_wp, CV_64FC1);
//    cv::Mat stddevs (m_hp, m_wp, CV_64FC1);
//    
//    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
//    {
//        Scalar mean, stddev;
//        meanStdDev(logLikelihoods.at(i,j), mean, stddev);
//        means.at<double>(i,j) = mean.val[0];
//        stddevs.at<double>(i,j) = stddev.val[0];
//    }
//    
//    GridMat znormLogLikelihoods(m_hp, m_wp);
//    
//    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
//    {
//        znormLogLikelihoods.set((logLikelihoods.at(i,j) - means.at<double>(i,j)) / stddevs.at<double>(i,j), i, j);
//        
//        cout << znormLogLikelihoods.get(i,j) << endl;
//    }
//    
//    GridMat probs (m_hp, m_wp);
//    
//    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
//    {
//        cv::Mat & c = znormLogLikelihoods.get(i,j);
//        cv::Mat p (c.rows, c.cols, CV_32FC1);
//        for (int k = 0; k < c.rows; k++)
//        {
//            p.at<float>(k,0) = (float) phi(c.at<double>(k,0));
//        }
//        
//        cout << p << endl;
//        
//        // Create an histogram for the cell region of blurred intensity values
//        int histSize[] = { (int) 10 };
//        int channels[] = { 0 }; // 1 channel, number 0
//        float tranges[] = { 0, 1 + 0.01 }; // thermal intensity values range: [0, 1)
//        const float* ranges[] = { tranges };
//
//        cv::Mat probsHist;
//        calcHist(&p, 1, channels, noArray(), probsHist, 1, histSize, ranges, true, false);
////        maxProbs.release();
//        cout << "hist (" << i << "," << j <<")" << probsHist << endl;
//    }
//     */
//
//    vector<Mat> outMasks;
//    vector<vector<Rect> > outBoundingRects;
//    
//    loadDataToMats   (m_DataPath + string("outputMasks/Thermal/"), "png", outMasks);
//	loadBoundingRects(m_DataPath + string("outputMasks/thermal_bounding_boxes_final.yml"), outBoundingRects);
//    //visualizeMasksWithinRects(outMasks, outBoundingRects); // DEBUG
//    
//    //
//    // < < < DEBUG > > >
//    // Testing step
//    // ------------
//    // (loading a special set of images for debugging purposes)
//    //
// /*
//    vector<Mat> testd;
//    vector<Mat> testmasksd;
//    vector< vector<Rect> > testrects;
//    vector< vector<int> > tags;
//    loadDebugTestData("../../Data", testd, testmasksd, testrects, tags);
//    
//    vector<GridMat> gframes_test;
//    vector<GridMat> gmasks_test;
//    grid(testd, testrects, tags, m_hp, m_wp, gframes_test);
//    grid(testmasksd, testrects, tags, m_hp, m_wp, gmasks_test);
//    
//    GridMat descriptors_test; // test data (descriptors)
////    trimodalFeatureExtractor.setThermalData(depthTestGrids, thermalTestMasks);
//
//	namedWindow("gframes_test preview");
//	for (int i = 0; i < gframes_test.size(); i++)
//	{
//		imshow("gframes_test preview", gframes_test[i].at(0,0));
//		waitKey(0);
//	}
//    thermalFeatureExtractor.setData(gframes_test, gmasks_test);
//    thermalFeatureExtractor.describe(descriptors_test);
//
//	// DEBUG: save to a file
//	descriptors_test.saveFS("test.yml");
//    
//	cout << "Predict clusters of test" << endl;
//    GridMat labels_test;
//    trimodalClusterer.predictClusters(descriptors_test, centers, labels_test); // using the centers computed in the training to predict the belonging
//    
////    GridMat classification, probabilities, logLikelihoods;
//    trimodalPixelClassificator.test(descriptors_test, labels_test, classification, probabilities, logLikelihoods);
//    
//    cout << classification << endl;
//    cout << probabilities << endl;
//	cout << logLikelihoods << endl;
//  */
//
}


/*
void TrimodalSegmentator::extractDepthFeatures()
{
    // Load data from disk: frames, masks, and rectangular bounding boxes
	vector<Mat> frames;
	vector<Mat> masks;
    vector< vector<Rect> > bounding_rects;
    vector< vector<int> > tags;
    
    loadDataToMats   (m_DepthFramesPath.c_str(), "jpg", frames);
	loadDataToMats   (m_DepthMasksPath.c_str(), "png", masks);
	loadBoundingRects(m_DepthBoundingRectsFilePath.c_str(), bounding_rects, tags);
    
    // Grid frames and masks
	vector<GridMat> gframes, gmasks;
    
	grid(frames, bounding_rects, m_hp, m_wp, gframes);
	grid(masks, bounding_rects , m_hp, m_wp, gmasks);
    
    // Divide into training and test
	const unsigned int n = gframes.size();
    const unsigned int nTest   = n * m_RateTest;
    const unsigned int nTrain  = n - nTest;
    
    RNG randgen(m_Seed);
    cv::Mat indices = shuffled(0, n-1, randgen);
    cv::Mat trainIndices (indices, Rect(0, 0, 1, nTrain));
    cv::Mat testIndices (indices, Rect(0, nTrain, 1, nTest));
    
	//
	// Pre-processing
	//
    
    vector<GridMat> gframes_train, gmasks_train; // the ones used to train
    
    select(gframes, trainIndices, gframes_train);
    select(gmasks, trainIndices, gmasks_train);
    
	//
	// Feature extraction
	//
	
    GridMat descriptions_train;
    
	DepthFeatureExtractor depthFeatureExtractor(m_hp, m_wp, m_DepthParam);
    depthFeatureExtractor.setData(gframes_train, gmasks_train);
    depthFeatureExtractor.describe(descriptions_train); // framews dimensionality reduced to the description dimensionality
	// DEBUG: save to a file
	descriptions_train.saveFS("train.yml");
    
	//
    // Clusterize people
    // -----------------
    // People are separated by "pose", i.e. each grid partition contains almost
    // the same information in all the images within a particular pose.
    //
    
    GridMat labels_train, centers; // Assigned cluster labels and cluster centers
  	
    TrimodalClusterer trimodalClusterer(m_NumClusters);
    trimodalClusterer.trainClusters(descriptions_train, labels_train, centers);
    
    //
    // Training step
    // -------------
    // Model the GMMs using the Expectation-Maximization.
    // Thus, having (hp * wp) * n_P * p modeled GMMs. Where (hp * wp) is the parametrized size of the grid.
    // And p is the number of poses.
    //
    
    TrimodalPixelClassifier trimodalPixelClassificator(m_NumMixtures);
    trimodalPixelClassificator.train(descriptions_train, labels_train, centers);
    
	cout << centers << endl;
	
	//
	// Train LogLikelihood PDFs
	//
    
    GridMat classification, probabilities, logLikelihoods;
    trimodalPixelClassificator.test(descriptions_train, labels_train, classification, probabilities, logLikelihoods);
    
    cout << classification << endl;
    cout << probabilities << endl;
	cout << logLikelihoods << endl;
    
    cv::Mat means (m_hp, m_wp, CV_64FC1);
    cv::Mat stddevs (m_hp, m_wp, CV_64FC1);
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        Scalar mean, stddev;
        meanStdDev(logLikelihoods.at(i,j), mean, stddev);
        means.at<double>(i,j) = mean.val[0];
        stddevs.at<double>(i,j) = stddev.val[0];
    }
    
    GridMat znormLogLikelihoods(m_hp, m_wp);
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        znormLogLikelihoods.set((logLikelihoods.at(i,j) - means.at<double>(i,j)) / stddevs.at<double>(i,j), i, j);
        
        cout << znormLogLikelihoods.get(i,j) << endl;
    }
    
    GridMat probs (m_hp, m_wp);
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat & c = znormLogLikelihoods.get(i,j);
        cv::Mat p (c.rows, c.cols, CV_32FC1);
        for (int k = 0; k < c.rows; k++)
        {
            p.at<float>(k,0) = (float) phi(c.at<double>(k,0));
        }
        
        cout << p << endl;
        
        // Create an histogram for the cell region of blurred intensity values
        int histSize[] = { (int) 10 };
        int channels[] = { 0 }; // 1 channel, number 0
        float tranges[] = { 0, 1 + 0.01 }; // thermal intensity values range: [0, 1)
        const float* ranges[] = { tranges };
        
        cv::Mat probsHist;
        calcHist(&p, 1, channels, noArray(), probsHist, 1, histSize, ranges, true, false);
        //        maxProbs.release();
        cout << "hist (" << i << "," << j <<")" << probsHist << endl;
    }
    
    
    //
    // < < < DEBUG > > >
    // Testing step
    // ------------
    // (loading a special set of images for debugging purposes)
    //
    
    vector<Mat> testd;
    vector<Mat> testmasksd;
    vector< vector<Rect> > testrects;
    loadDebugTestData("../../Data", testd, testmasksd, testrects);
    
    vector<GridMat> gframes_test;
    vector<GridMat> gmasks_test;
    grid(testd, testrects, m_hp, m_wp, gframes_test);
    grid(testmasksd, testrects, m_hp, m_wp, gmasks_test);
    
    GridMat descriptors_test; // test data (descriptors)
    //    trimodalFeatureExtractor.setThermalData(depthTestGrids, thermalTestMasks);
    
	namedWindow("gframes_test preview");
	for (int i = 0; i < gframes_test.size(); i++)
	{
		imshow("gframes_test preview", gframes_test[i].at(0,0));
		waitKey(0);
	}
    thermalFeatureExtractor.setData(gframes_test, gmasks_test);
    thermalFeatureExtractor.describe(descriptors_test);
    
	// DEBUG: save to a file
	descriptors_test.saveFS("test.yml");
    
	cout << "Predict clusters of test" << endl;
    GridMat labels_test;
    trimodalClusterer.predictClusters(descriptors_test, centers, labels_test); // using the centers computed in the training to predict the belonging
    
    //    GridMat classification, probabilities, logLikelihoods;
    trimodalPixelClassificator.test(descriptors_test, labels_test, classification, probabilities, logLikelihoods);
    
    cout << classification << endl;
    cout << probabilities << endl;
	cout << logLikelihoods << endl;
}
*/

//void TrimodalSegmentator::exec()
//{	
//    //
//    // Pre-processing
//    // -------------
//    //
//    // Trim people (bounding boxes) from frames, and grid them on (hp x wp)-sized cells
//    //
//    
//    vector<GridMat> depthGrids, thermalGrids;
//    vector<GridMat> depthMasks, thermalMasks;
//
//    grid(m_DepthFrames, m_BoundingRects, m_hp, m_wp, depthGrids);
//    grid(m_ThermalFrames, m_BoundingRects, m_hp, m_wp, thermalGrids);
//    
//    grid(m_DepthMasks, m_BoundingRects, m_hp, m_wp, depthMasks);
//    grid(m_ThermalMasks, m_BoundingRects, m_hp, m_wp, thermalMasks);
//    
//    
//
//    
//    // The not-so-right way to divide it (for debuggin purposes)
//    
//    //    cv::Mat trainIndices (8,1, DataType<int>::type);
//    //    cv::Mat testIndices (2,1, DataType<int>::type);
//    //
//    //    trainIndices.at<int>(0,0) = 1;
//    //    trainIndices.at<int>(1,0) = 2;
//    //    trainIndices.at<int>(2,0) = 3;
//    //    trainIndices.at<int>(3,0) = 4;
//    //    trainIndices.at<int>(4,0) = 5;
//    //    trainIndices.at<int>(5,0) = 7;
//    //    trainIndices.at<int>(6,0) = 8;
//    //    trainIndices.at<int>(7,0) = 9;
//    //
//    //    testIndices.at<int>(0,0) = 0;
//    //    testIndices.at<int>(1,0) = 6;
//    
//    // Select the training set
//    
//    vector<GridMat> depthTrainGrids, thermalTrainGrids;
//    vector<GridMat> depthTrainMasks, thermalTrainMasks;
//    
//    select(depthGrids, trainIndices, depthTrainGrids);
//    select(thermalGrids, trainIndices, thermalTrainGrids);
//    
//    select(depthMasks, trainIndices, depthTrainMasks);
//    select(thermalMasks, trainIndices, thermalTrainMasks);
//    
//    
//    //
//    // Describe cells
//    // --------------
//    // Descriptors computed will be be used in a clusterization (pre-classification) and in classification
//    //
//    
//    TrimodalFeatureExtractor trimodalFeatureExtractor(m_hp, m_hp);
//    
//    trimodalFeatureExtractor.setThermalParam(m_ThermalParam);
//    trimodalFeatureExtractor.setDepthData(depthTrainGrids, depthTrainMasks);
//    trimodalFeatureExtractor.setThermalData(thermalTrainGrids, thermalTrainMasks);
//    
//    GridMat trainThermal;
//    GridMat trainDepth;
//    trimodalFeatureExtractor.describe(trainThermal, trainDepth);
//
//    
//    //
//    // Clusterize people
//    // -----------------
//    // People are somehow separated by pose, in such a way each grid partition contains almost
//    // the same information in all the images within a particular pose.
//    //
//    
//    GridMat trainLabels, centers; // Assigned cluster labels and cluster centers
//    
//    TrimodalClusterer trimodalClusterer(m_NumClusters);
//    trimodalClusterer.trainClusters(trainThermal, trainLabels, centers);
//    
//    
//    //
//    // Feature extraction
//    // ------------------
//    // In the grid cell G(i,j) throughout the n_P images of pose P, extract features to then, model
//    // GMM.
//    //
//    
//    unsigned int classes = 3; // hardcoded since it is a typical value
//    TrimodalPixelClassifier trimodalPixelClassificator(classes);
//
//    
//    //
//    // Training step
//    // -------------
//    // Model the GMMs using the Expectation-Maximization.
//    // Thus, having (hp * wp) * n_P * p modeled GMMs. Where (hp * wp) is the parametrized size of the grid.
//    // And p is the number of poses.
//    //
//    
//    trimodalPixelClassificator.train(trainThermal, trainLabels, centers);
//
//    
//    //
//    // Testing step
//    // ------------
//    // -
//    //
//        
////    vector<GridMat> thermalTestGrids;
////    vector<GridMat> thermalTestMasks;
////    select(thermalGrids, testIndices, thermalTestGrids);
////    select(thermalMasks, testIndices, thermalTestMasks);
////    
////    GridMat test; // test data (descriptors)
////    trimodaFeatureExtractor.setThermalData(thermalTestGrids, thermalTestMasks);
////    trimodaFeatureExtractor.describe(test);
////    
////    GridMat testLabels;
////    trimodalClusterer.predictClusters(test, centers, testLabels); // using the centers computed in the training to predict the belonging
////    
////    GridMat classification, probabilities;
////    trimodalPixelClassificator.test(test, testLabels, classification, probabilities);
////    
////    cout << classification << endl;
////    cout << probabilities << endl;
//    
//    //
//    // < < < DEBUG > > >
//    // Testing step
//    // ------------
//    // (loading a special set of images for debugging purposes)
//    //
//    
//    vector<Mat> testd;
//    vector<Mat> testmasksd;
//    vector< vector<Rect> > testrects;
//    loadDebugTestData("Data", testd, testmasksd, testrects);
//    
//    vector<GridMat> thermalTestGrids;
//    vector<GridMat> thermalTestMasks;
//    grid(testd, testrects, m_hp, m_wp, thermalTestGrids);
//    grid(testmasksd, testrects, m_hp, m_wp, thermalTestMasks);
//    
//    GridMat test; // test data (descriptors)
////    trimodalFeatureExtractor.setThermalData(depthTestGrids, thermalTestMasks);
//    trimodalFeatureExtractor.setThermalData(thermalTestGrids, thermalTestMasks);
//    trimodalFeatureExtractor.describe(test, test);
//    
//    GridMat testLabels;
//    trimodalClusterer.predictClusters(test, centers, testLabels); // using the centers computed in the training to predict the belonging
//    
//    GridMat classification, probabilities;
//    trimodalPixelClassificator.test(test, testLabels, classification, probabilities);
//    
//    cout << classification << endl;
//    cout << probabilities << endl;
//}


/*
 * Trim subimages, defined by rects (bounding boxes), from image frames
 */
void TrimodalSegmentator::grid(vector<cv::Mat> frames, vector< vector<cv::Rect> > rects, vector< vector<int> > tags, unsigned int crows, unsigned int ccols, vector<GridMat> & grids)
{
    //namedWindow("grided subject");
    // Seek in each frame ..
    for (unsigned int f = 0; f < rects.size(); f++)
    {
        // .. all the people appearing
        for (unsigned int r = 0; r < rects[f].size(); r++)
        {
            if (rects[f][r].height >= m_hp && rects[f][r].width >= m_wp)
            {
                cv::Mat subject (frames[f], rects[f][r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject = (subject == (m_OffsetID + r));
                subject.release();
                
                GridMat g (maskedSubject, crows, ccols);
                grids.push_back( g );
            }
        }
    }
}

/*
 *    < < <   DEBUG   > > >
 */
void TrimodalSegmentator::loadDebugTestData(const char* path, vector<cv::Mat> & test, vector<cv::Mat> & testmasks, vector< vector<cv::Rect> > & testrects)
{
    if (path == NULL)
    {
        cerr << "Error: need to provide a dir with frames to read" << endl;
        return;
    }
    
    string spath = string(path).append("/"); // "const char*" to "string" plus the "/" at the end
    
    vector< vector<int> > tags; // not used here
    string bbPath = spath + "testBoundingBoxes.txt";
    loadBoundingRects (bbPath.c_str(), testrects, tags);
    
    // Load the image frames first
    test.clear();
    string tPath = spath + "Test";
    loadDataToMats  (tPath.c_str(), "jpg", test);
    
    testmasks.clear();
    string masksPath = spath + "TestMasks";
    loadDataToMats  (masksPath.c_str(), "png", testmasks);
}

/*
 * Load all the data frames (color, depth, and thermal frames) and rects
 */
//void TrimodalSegmentator::loadAllData(const char* colorPath, const char* depthPath, const char* thermalPath, const char* depthMasksPath, const char* thermalMasksPath, const char* rectsPath)
//{
//    if (colorPath == NULL || depthPath == NULL || thermalPath == NULL
//        || depthMasksPath == NULL || thermalMasksPath == NULL || rectsPath == NULL)
//    {
//        cerr << "Error: need to provide actual paths to the requiere data." << endl;        
//        return;
//    }
//      
//    string bbPath = string(rectsPath);
//    loadBoundingRects (bbPath.c_str(), m_BoundingRects);
//    
////    // Load the image frames first
////    // TODO: Fill and discomment
//    
////    string cPath = string(colorPath);
////    m_ColorFrames.clear();
////    loadDataFrames  (cPath.c_str(), "jpg", m_ColorFrames);
////    
//    string dPath = string(depthPath);
//    m_DepthFrames.clear();
//    loadDataFrames  (dPath.c_str(), "png", m_DepthFrames);
//    
//    m_ThermalFrames.clear();
//    string tPath = string(thermalPath);
//    loadDataFrames  (tPath.c_str(), "jpg", m_ThermalFrames);
//    
//    m_DepthMasks.clear();
//    string mPath = string(depthMasksPath);
//    loadDataFrames  (mPath.c_str(), "png", m_DepthMasks);
//    
//    m_ThermalMasks.clear();
//    mPath = string(thermalMasksPath);
//    loadDataFrames  (mPath.c_str(), "png", m_ThermalMasks);
//    
////    // DEBUG
////    // Load the bounding boxes, in near range extracted from the depth
////
////    vector<Rect> ra;
////    ra.push_back(Rect(542, 250, 584-542, 305-250));
////    m_PeopleRects.push_back(ra);
////    
////    vector<Rect> r0;
////    r0.push_back(Rect(94, 85, 277-94, 479-85));
////    m_PeopleRects.push_back(r0);
////    
////    vector<Rect> r1;
////    r1.push_back(Rect(198, 68, 378-198, 479-68));
////    m_PeopleRects.push_back(r1);
////    
////    vector<Rect> r2;
////    r2.push_back(Rect(255, 191, 479-255, 479-191));
////    m_PeopleRects.push_back(r2);
////    
////    vector<Rect> r3;
////    r3.push_back(Rect(288, 192, 469-288, 479-192));
////    m_PeopleRects.push_back(r3);
////    
////    vector<Rect> r4;
////    r4.push_back(Rect(293, 199, 487-293, 479-199));
////    m_PeopleRects.push_back(r4);
////    
////    vector<Rect> rb;
////    rb.push_back(Rect(429, 302, 509-429, 326-302));
////    m_PeopleRects.push_back(rb);
////    
////    vector<Rect> r5;
////    r5.push_back(Rect(154, 38, 317-154, 479-38));
////    m_PeopleRects.push_back(r5);
////    
////    vector<Rect> r6;
////    r6.push_back(Rect(287, 201, 465-287, 479-201));
////    m_PeopleRects.push_back(r6);
////    
////    vector<Rect> r7;
////    r7.push_back(Rect(127, 107, 289-127, 479-107));
////    m_PeopleRects.push_back(r7);
//}


/*
 * Load data frames
 */
//void TrimodalSegmentator::loadDataFrames(const char* path, const char* format, vector<Mat> & frames)
//{
//    DIR *dp;
//    struct dirent *dirp;
//    struct stat filestat;
//    
//    dp = opendir(path);
//    if (dp == NULL)
//    {
//        cerr << "Error: opening file (" << path << ")" << endl;
//        return;
//    }
//    
//    string filepath, filename;
//    while ( (dirp = readdir(dp)) )
//    {
//        filename = string(dirp->d_name);
//        filepath = string(path) + "/" + dirp->d_name;
//        
//        // If the file is a directory (or is in some way invalid) we'll skip it
//        if (stat( filepath.c_str(), &filestat )) continue;
//        if (S_ISDIR( filestat.st_mode ))         continue;
//        if (filename.compare(".DS_Store") == 0)  continue;
//        
//        cv::Mat img = imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
//        frames.push_back(img);
//    }
//    
//    closedir(dp);
//}

/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void TrimodalSegmentator::loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames)
{
    const char* path = dir.c_str();
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && iter->path().extension().string().compare(".DS_Store") != 0 && iter->path().filename().string().compare("renamer.py") != 0)
			{
                //std::cout << iter->path().string() << std::endl;
				cv::Mat img = cv::imread( iter->path().string(), CV_LOAD_IMAGE_ANYDEPTH );
				frames.push_back(img);
			}
		}
	}
}

/*
 * Load the people data (bounding boxes coordinates)
 */
void TrimodalSegmentator::loadBoundingRects(string file, vector< vector<cv::Rect> > & rects, vector< vector<int> >& tags)
{
    cv::FileStorage fs;
    fs.open(file.c_str(), cv::FileStorage::READ);

    int num_frames;
    fs["num_frames"] >> num_frames;
    
    for (int i = 0; i < num_frames; i++)
    {
        stringstream ss;
        ss << i;

        std::vector<int> v, w;
        fs[string("coords_") + ss.str()] >> v;
        fs[string("tags_") + ss.str()] >> w;
        /*
        for (int j = 0; j < v.size(); j++)
            cout << v[j] << ",";
        cout << endl; */        
        vector<cv::Rect> frame_rects;
        for (int j = 0; j < v.size() / 4; j++)
        {
            int x0 = v[j*4];
            int y0 = v[j*4+1];
            int x1 = v[j*4+2];
            int y1 = v[j*4+3];
            
            frame_rects.push_back( cv::Rect(x0, y0, x1 - x0, y1 - y0) );
        }
        //cout << i << " rects: " << frame_rects.size() << endl;
        rects.push_back(frame_rects);
        tags.push_back(w);
    }

//    for (int i = 0; i < num_frames; i++)
//    {
//        stringstream ss;
//        string s;
//        ss << i;
//        s = ss.str();
//        string a = "tags_";
//        a+=s;
//
//        std::vector<int> v;
//        fs[a] >> v;
//        
//        bbTags.push_back(v);
//    }
    
    fs.release();
    
    /*
    std::ifstream in(path);
    string line;

    // Read lines
    while( getline(in, line) )
    {
//        // DEBUG
//        line = "0#";
//        line = "1#1 2 -10 2#";
//        line = "2#1 2 -10 2#3 4 11 0#";
        
        // First number in line is the number of bounding rects in the frame
        unsigned long inipos = line.find("#");
        string head = line.substr(0, inipos);
        istringstream buffer(head);
        int numOfRects;
        buffer >> numOfRects;
        
        // The rest are the coordinates of each rect in a line
        string tail (line.substr(inipos));
        inipos = inipos - 1; // a detail (for further processing)
        
        // Extract the coordinates separating by the delimiters and create the rects
        vector<Rect> frameRects;
        unsigned long endpos;
        for (int i = 0; i < numOfRects; i++)
        {
            string aux = tail.substr(inipos + 1);   // Cut in the left bound
            endpos = aux.find("#");                 // Seek the right bound delimiter
            string s = aux.substr(0,endpos);        // Cut in the right bound       
            inipos = endpos + 1;
            
            int x0, y0, x1, y1;
            sscanf(s.c_str(), "%d %d %d %d", &x0, &y0, &x1, &y1);
            frameRects.push_back(Rect(x0, y0, x1-x0, y1-y0));
        }
        
        rects.push_back(frameRects);
    }
     */
}

cv::Mat TrimodalSegmentator::shuffled(int a, int b, cv::RNG randGen)
{
    cv::Mat vec (b-a+1, 1, cv::DataType<int>::type);
    for (int i = a; i <= b; i++)
    {
        vec.at<int>(i-a, 0) = i;
    }
    
    randShuffle(vec, 1, &randGen);
    
    return vec;
}

void TrimodalSegmentator::select(vector<GridMat> grids, vector<int> indices, vector<GridMat> & selection)
{
    selection.resize(indices.size());
    for (int i = 0; i < indices.size(); i++)
    {
        selection[i] = grids[indices[i]];
    }
}

void TrimodalSegmentator::setDataPath(string dataPath)
{
    m_DataPath = dataPath;
    
    const char* path = m_DataPath.c_str();
	if( exists( path ) )
	{
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( is_directory( *iter ) )
			{
                string scenePath = iter->path().string();
				m_ScenesPaths.push_back(scenePath);
                
                cout << "Scene found: " << scenePath << endl;
			}
		}
	}
    else
    {
        cerr << "Data path is not containing any scene(s)!" << endl;
    }
}