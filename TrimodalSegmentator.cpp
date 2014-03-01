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


using namespace std;

/*
 * Constructor
 */

TrimodalSegmentator::TrimodalSegmentator(const unsigned int hp, const unsigned int wp, const unsigned char offsetID)
    : m_hp(hp), m_wp(wp), m_OffsetID(offsetID)
{ }


void TrimodalSegmentator::extractColorFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const ColorParametrization param, GridMat& descriptors)
{
    ColorFeatureExtractor fe(m_hp, m_wp, param);
    
    fe.describe(gframes, gmasks, descriptors);
}


void TrimodalSegmentator::extractMotionFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const MotionParametrization param, GridMat& descriptors)
{
    MotionFeatureExtractor fe(m_hp, m_wp, param);
    
    fe.describe(gframes, gmasks, descriptors);
}


void TrimodalSegmentator::extractDepthFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const DepthParametrization param, GridMat& descriptors)
{
    DepthFeatureExtractor fe(m_hp, m_wp, param);
    
    fe.describe(gframes, gmasks, descriptors);
}


void TrimodalSegmentator::extractThermalFeatures(std::vector<GridMat>& gframes, std::vector<GridMat>& gmasks, const ThermalParametrization param, GridMat& descriptors)
{
    ThermalFeatureExtractor fe(m_hp, m_wp, param);
    
    fe.describe(gframes, gmasks, descriptors);
}


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
//}


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
