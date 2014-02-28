//
//  TrimodalPixelClassifier.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 21/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "TrimodalPixelClassifier.h"
#include "GridMat.h"

#include <opencv2/opencv.hpp>

//#include "EM.h"

#include <vector>
#include <fstream>

#define __PI 3.14159265

using namespace std;

TrimodalPixelClassifier::TrimodalPixelClassifier(unsigned int classes) : m_classes(classes)
{ }

//void TrimodalPixelClassifier::preprocessThermalTrainData(vector<GridMat> tGridMats, vector<GridMat> masksGridMats, cv::Mat indices, int numClusters, vector<Mat> cellsClusterLabels, vector<GridMat> & tTrain)
//{
//    /*
//     * The cell grids had been already clusterized (the cells in the same image grid can have different labels).
//     * For each cell, we prepare the data to train different GMMs, one for each cluster in the cell (i,j).
//     */
//    tTrain.resize(numClusters);
//    for (int p = 0; p < numClusters; p++)
//        tTrain[p] = GridMat( tGridMats[0].crows(), tGridMats[0].ccols() );
//    
//    for (int k = 0; k < indices.rows; k++)
//    {
//        int idx = indices.at<int>(k,0);
//        GridMat & tGridMat = tGridMats[idx];
//        GridMat & mask = masksGridMats[idx];
//        
//        cout << "Describing thermal image " << k << "/" << tGridMats.size() - 1 << " ..." << endl;
//    
//        for (int i = 0; i < tGridMat.crows(); i++) for (int j = 0; j < tGridMat.ccols(); j++)
//        {
//            int clusterLabel =  cellsClusterLabels[k].at<int>(i,j);
//            
//            cv::Mat tDescriptorsMatrix;
//            describeThermalCell(tGridMat.at(i,j), mask.at(i,j), tDescriptorsMatrix);
//            
//            tTrain[clusterLabel].vconcat(tDescriptorsMatrix, i, j);
//        }
//    }
//}

//void TrimodalPixelClassifier::preprocessThermalTestInstance(GridMat & tGridMat, GridMat & mask, GridMat & tTest)
//{
//    for (int i = 0; i < tGridMat.crows(); i++) for (int j = 0; j < tGridMat.ccols(); j++)
//    {
//        cv::Mat tDescriptorsMatrix;
//        describeThermalCell(tGridMat.at(i,j), mask.at(i,j), tDescriptorsMatrix);
//        tTest.vconcat(tDescriptorsMatrix, i, j);
//        tDescriptorsMatrix.release();
//    }
//}
//
//void TrimodalPixelClassifier::describeThermalCell(const cv::Mat & c, const cv::Mat & m, cv::Mat & tDescriptorsMatrix)
//{
//    //
//    // Extend the cell borders or mask borders to properly convolve a patch
//    //
//    
////    int psize; // patch side size
////    int cpsize; // corrected patch side size (odd)
////    int margin; // distance from the center of kernel to a bound (x or y-direction)
////    
////    psize = min(c.rows, c.cols) * m_kparam;
////    cpsize = psize - (psize % 2) + 1;
////    margin = (cpsize - 1) / 2;
//    
////    cv::Mat cell, mask;
////    
////    copyMakeBorder(c, cell, margin, margin, margin, margin, BORDER_CONSTANT, 0);
////    copyMakeBorder(m, mask, margin, margin, margin, margin, BORDER_CONSTANT, 0);
//
//    //
//    // Compute thermal intensities histograms in patches
//    //
//    
////    for (int i = margin; i < cell.rows - margin; i++) for (int j = margin; j < cell.cols - margin; j++)
////    {
////        if (mask.at<uchar>(i,j) == 0) continue;
//////        cout << Rect(i-d, j-d, cpsize, cpsize) << endl;
////        cv::Mat patch = cv::Mat(cell, Rect(j-margin, i-margin, cpsize, cpsize)); // Rect(x,y), x are columns
////        cv::Mat maskPatch = cv::Mat(mask, Rect(j-margin, i-margin, cpsize, cpsize));
////        
////        // Create an histogram for the cell region of blurred intensity values
////        int histSize[] = { (int) m_cellbins };
////        int channels[] = { 0 }; // 1 channel, number 0
////        float tranges[] = { 0, 256 }; // thermal intensity values range: [0, 256)
////        const float* ranges[] = { tranges };
////        
////        cv::Mat tHist;
////        calcHist(&patch, 1, channels, maskPatch, tHist, 1, histSize, ranges, true, false);
////        patch.release();
////        maskPatch.release();
////        
////        double minVal, maxVal;
////        minMaxLoc(tHist, &minVal, &maxVal);
////
////        cv::Mat tHistNorm ((tHist - minVal) / (maxVal - minVal));
////        transpose(tHistNorm, tHist);
////        tHistNorm.release();
////        
////        // Concatenate the cell region histrogram to the grid descriptor
////        if (tDescriptorsMatrix.rows == 0 && tDescriptorsMatrix.cols == 0) tDescriptorsMatrix = tHist;
////        else vconcat(tDescriptorsMatrix, tHist, tDescriptorsMatrix);
////        
////        tHist.release();
////    }
//    
////    for (int i = margin; i < cell.rows - margin; i++) for (int j = margin; j < cell.cols - margin; j++)
////    {
////        if (mask.at<uchar>(i,j) == 0) continue;
////        //        cout << Rect(i-d, j-d, cpsize, cpsize) << endl;
////        cv::Mat patch = cv::Mat(cell, Rect(j-margin, i-margin, cpsize, cpsize)); // Rect(x,y), x are columns
////        cv::Mat maskPatch = cv::Mat(mask, Rect(j-margin, i-margin, cpsize, cpsize));
//    
//        // Create an histogram for the cell region of blurred intensity values
//        int histSize[] = { (int) m_cellbins };
//        int channels[] = { 0 }; // 1 channel, number 0
//        float tranges[] = { 0, 256 }; // thermal intensity values range: [0, 256)
//        const float* ranges[] = { tranges };
//        
//        cv::Mat tHist;
//        calcHist(&c, 1, channels, m, tHist, 1, histSize, ranges, true, false);
////        patch.release();
////        maskPatch.release();
//    
//        double minVal, maxVal;
//        minMaxLoc(tHist, &minVal, &maxVal);
//        
////        cv::Mat tHistNorm ((tHist - minVal) / (maxVal - minVal));
//        cv::Mat tHistNorm (tHist / sum(tHist).val[0]);
//        transpose(tHistNorm, tHist);
//        tHistNorm.release();
//        
//        // Concatenate the cell region histrogram to the grid descriptor
//        if (tDescriptorsMatrix.rows == 0 && tDescriptorsMatrix.cols == 0) tDescriptorsMatrix = tHist;
//        else vconcat(tDescriptorsMatrix, tHist, tDescriptorsMatrix);
//        
//        tHist.release();
////    }
//}

void TrimodalPixelClassifier::train(GridMat descriptions, GridMat labels, GridMat centers)
{
    // Set for further use some variables
    m_TrainDescriptions = descriptions;
    m_TrainLabels       = labels;
    m_TrainCenters      = centers;
    
    //
    // Perform the training
    //
    
    int hp = m_TrainDescriptions.crows();
    int wp = m_TrainDescriptions.ccols();
    int C = m_TrainCenters.at(0,0).rows; // number of clusters in the previous clusterization
    
    m_TEMMatrix.clear();
    m_TEMMatrix.resize(C); // The GMMs of all the poses
    
    for (int k = 0; k < m_TEMMatrix.size(); k++)
    {
        vector<cv::EM> tEMArray; // The GMMs of a pose
        tEMArray.resize(hp * wp);
        
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            cv::Mat t;
            for (int d = 0; d < m_TrainDescriptions.at(i,j).rows; d++) //(0,0).rows; d++)
            {
                if (k == m_TrainLabels.at(i,j).at<int>(d,0))
                {
                    if (t.empty()) t = m_TrainDescriptions.at(i,j).row(d);
                    else vconcat(t, m_TrainDescriptions.at(i,j).row(d), t);
                }
            }
            
            
            cout << "Training cluster " << k << ", GMM (" << i << ","  << j << ") ..." << endl;
            
			
			cv::TermCriteria term;//(CV_TERMCRIT_ITER, 10000000, 0.000001);
            cv::EM em(3, cv::EM::COV_MAT_GENERIC); // The GMM of a pose grid's cell
            
			cv::Mat train_loglikelihoods, train_probs;
            bool success = em.train(t, train_loglikelihoods, cv::noArray(), train_probs);
            t.release();
            cv::FileStorage fs;
			fs.open("likelihoods.yml", cv::FileStorage::WRITE);
			fs << "likelihoods" << train_loglikelihoods;
			//cout << train_probs << endl;

			//std::stringstream ss;
			//ss << k << i << j;
			//std::ofstream fout;
			//fout.open(ss.str());

			//Mat w = em.get<Mat>("weights");
			//
			//cout << "weights: " << cv::sum(w).val[0] << endl;
			//Mat m = em.get<Mat>("means");
			//ss.clear();
			//ss << cv::format(m, "csv") << endl;

			//fout << ss << endl; 
			//fout.close();

			//cout << m << endl;
            
            tEMArray[i * wp + j] = em;
        }
        m_TEMMatrix[k] = tEMArray;
    }

	modelLogLikelihoodsPDFs(descriptions, labels);
}

//void TrimodalPixelClassifier::train(vector<GridMat> tGridsTrain)
//{
//    m_TEMMatrix.resize(tGridsTrain.size()); // The GMMs of all the poses
//    
//    for (int k = 0; k < tGridsTrain.size(); k++)
//    {
//        vector<EM> tEMArray; // The GMMs of a pose
//        tEMArray.resize( tGridsTrain[k].crows() * tGridsTrain[k].ccols() );
//        
//        unsigned int hp = tGridsTrain[k].crows();
//        unsigned int wp = tGridsTrain[k].ccols();
//        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//        {
//            cout << "Training cluster " << k << ", GMM (" << i << ","  << j << ") ..." << endl;
//            
//            EM::EM em(m_patchclusters); // The GMM of a pose grid's cell
////            cv::Mat cellTrain;
////            transpose(tGridsTrain[k].at(i,j), cellTrain);
//            bool success = em.train(tGridsTrain[k].at(i,j));
//            cout << success << " " << em.isTrained() << endl;
////            cellTrain.release();
//            
//            tEMArray[i * wp + j] = em;
//        }
//        m_TEMMatrix[k] = tEMArray;
//    }
//}

void TrimodalPixelClassifier::modelLogLikelihoodsPDFs(const GridMat descriptors, const GridMat labels)
{
	// Attention! The label indicates the cluster in the pre-clusterization, is not referring to "subject" or "not subject".
	 
	vector< vector<GridMat> > clustersLogLikelihoods;
	clustersLogLikelihoods.resize(m_TEMMatrix.size());
	for (int i = 0; i < clustersLogLikelihoods.size(); i++)
	{
		clustersLogLikelihoods[i].resize(m_classes);
		for (int j = 0; j < clustersLogLikelihoods[i].size(); j++)
		{
			clustersLogLikelihoods[i][j].create(descriptors.crows(), descriptors.ccols());
		}
	}
	

	for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {
		for (int d = 0; d < descriptors.at(i,j).rows; d++)
		{          
			int clusterIdx = labels.at(i,j).at<int>(d,0);
            const cv::EM & em = m_TEMMatrix[clusterIdx][i * descriptors.ccols() + j];

            cv::Mat probs, likes;
//            Vec2d prediction = em.predict( descriptors.at(i,j).row(d), probs, likes );

            cv::Vec2d prediction = em.predict( descriptors.at(i,j).row(d), probs );

			cv::Mat_<double> logLikelihood (1,1);
			logLikelihood.at<double>(0,0) = prediction.val[0];
			//clustersLogLikelihoods[clusterIdx][prediction.val[1]].vconcat(logLikelihood, i, j); //TODO: fix
		}
	}


	
	clustersLogLikelihoods[0][0].saveFS("loglikes_00");

	std::cout << std::endl;
}

void TrimodalPixelClassifier::test(GridMat descriptors, GridMat labels, GridMat & classes, 
	GridMat & probabilities, GridMat & logLikelihoods)
{
    classes.create(descriptors.crows(), descriptors.ccols());
    probabilities.create(descriptors.crows(), descriptors.ccols());
	logLikelihoods.create(descriptors.crows(), descriptors.ccols());
    
    for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {
		cout << endl;
        cv::Mat C = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<int>::type);
        cv::Mat P = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<double>::type);
        cv::Mat L = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<double>::type);
        
        for (int d = 0; d < descriptors.at(i,j).rows; d++)
        {
            cout << m_TEMMatrix.size() << endl;
            
            const cv::EM & em = m_TEMMatrix[labels.at(i,j).at<int>(d,0)][i * descriptors.ccols() + j];

            cv::Mat probs, likes;
//            Vec2d prediction = em.predict( descriptors.at(i,j).row(d), probs, likes );
            cv::Vec2d prediction = em.predict( descriptors.at(i,j).row(d), probs );
            C.at<int>(d,0) = prediction.val[1];
            P.at<double>(d,0) = probs.at<double>(0, prediction.val[1]);
			L.at<double>(d,0) = likes.at<double>(0, prediction.val[1]);

			cout << prediction.val[0] << endl;
			cout << probs << endl;
			cout << likes << endl;
        }
        
        classes.set(C, i, j);
        probabilities.set(P, i, j);
		logLikelihoods.set(L, i, j);
    }
}

void TrimodalPixelClassifier::test2(GridMat descriptors, GridMat labels, GridMat & classes,
                                   GridMat & probabilities, GridMat & logLikelihoods)
{
    int k = descriptors.at(0, 0).cols;
    
    classes.create(descriptors.crows(), descriptors.ccols());
    probabilities.create(descriptors.crows(), descriptors.ccols());
	logLikelihoods.create(descriptors.crows(), descriptors.ccols());
    
    for (int i = 0; i < descriptors.crows(); i++) for (int j = 0; j < descriptors.ccols(); j++)
    {
		cout << endl;
        cv::Mat C = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<int>::type);
        cv::Mat P = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<double>::type);
        cv::Mat L = cv::Mat(descriptors.at(i,j).rows, 1, cv::DataType<double>::type);
        
        for (int d = 0; d < descriptors.at(i,j).rows; d++)
        {
            cout << m_TEMMatrix.size() << endl;
            
            const cv::EM & em = m_TEMMatrix[labels.at(i,j).at<int>(d,0)][i * descriptors.ccols() + j];
            
//            cv::Mat probs, likes;
//            Vec2d prediction = em.predict( descriptors.at(i,j).row(d), probs, likes );
//            C.at<int>(d,0) = prediction.val[1];
//            P.at<double>(d,0) = probs.at<double>(0, prediction.val[1]);
//			L.at<double>(d,0) = likes.at<double>(0, prediction.val[1]);
//            
//			cout << prediction.val[0] << endl;
//			cout << probs << endl;
//			cout << likes << endl;
            
//            printf("%d, %.4f, %.6f\n", C.at<int>(d,0), prediction.val[0], P.at<double>(d,0));
            
			int nclusters = em.get<int>("nclusters");
            cv::Mat probs = cv::Mat::zeros(1, nclusters, cv::DataType<int>::type);
            
            cv::Mat x = descriptors.at(i,j).row(d);
			cout << x.type() << endl;
            
            cv::Mat weights = em.get<cv::Mat>("weights");
            cv::Mat means = em.get<cv::Mat>("means");
			vector<cv::Mat> covs = em.get<vector<cv::Mat> >("covs");
            
            double maxProb = std::numeric_limits<double>::min();
			for (int m = 0; m < nclusters; m++)
			{
                cv::Mat diff, dblx;
				x.convertTo(dblx, CV_64F);
				cv::subtract(dblx, means.row(m), diff);
            
                cv::Mat diffT, invcovs;
				transpose(diff, diffT);
				invert(covs[m], invcovs);
                cv::Mat prod = (diff * invcovs) * diffT;
			
                double mahalDist = /*weights.at<double>(0,m) **/ exp(-0.5 * prod.at<double>(0,0));
                double prob = ( 1.0 / sqrt(pow(2.0*__PI, k)) * determinant(covs[m]) ) * mahalDist;
				//cout << "prob: " << prob << endl;
                if (prob > maxProb) maxProb = prob;
			}
            P.at<double>(d,0) = maxProb;
        }
        
        classes.set(C, i, j);
        probabilities.set(P, i, j);
		logLikelihoods.set(L, i, j);
    }
}

//void TrimodalPixelClassifier::predict(GridMat & grid, GridMat & mask, cv::Mat cellsLabels, GridMat & cellsFeatures)
//{
//    namedWindow("cell");
//    const unsigned int hp = cellsLabels.rows; // Get the 0-indexed attributes, they are all the same
//    const unsigned int wp = cellsLabels.cols;
//
//    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
//    {
//        imshow("cell", grid.at(i,j));
//        waitKey(0);
//        int cellLabel = cellsLabels.at<int>(i,j);
//        EM & em = m_TEMMatrix[cellLabel][i * wp + j];
//        
//        cv::Mat maxProbs ( cellsFeatures.at(i,j).rows, 1, DataType<float>::type );
//        cv::Mat classes ( cellsFeatures.at(i,j).rows, 1, DataType<uchar>::type );
//        for (int k = 0; k < cellsFeatures.at(i,j).rows; k++)
//        {
//            cv::Mat probs;
//               
//            Vec2d prediction = em.predict( cellsFeatures.at(i,j).row(k), probs );
//            
//            cout << probs << endl;
//            cout << cellsFeatures.at(i,j).row(k) << endl;
//            cout << "prediction " << prediction << endl;
//            cout << prediction.val[1] << endl;
//            maxProbs.at<float>(k,0) = (float) probs.at<double>(0, prediction.val[1]);
//            classes.at<uchar>(k,0) = static_cast<unsigned char>(prediction.val[1]);
//            cout << maxProbs.at<float>(k,0) << endl;
//            
//        }
//        
//        cout << "GMM(" << i << "," << j << ")" << endl;
//        
//        // Create an histogram for the cell region of blurred intensity values
//        int histSize[] = { (int) 10 };
//        int channels[] = { 0 }; // 1 channel, number 0
//        float tranges[] = { 0, 1 + 0.01 }; // thermal intensity values range: [0, 1)
//        const float* ranges[] = { tranges };
//        
//        cv::Mat probsHist;
//        calcHist(&maxProbs, 1, channels, noArray(), probsHist, 1, histSize, ranges, true, false);
//        maxProbs.release();
//        
//        cout << "Probability distribution [0-1]: " << probsHist << endl;
//        
//        // Create an histogram for the cell region of blurred intensity values
//        int ahistSize[] = { static_cast<int>(m_patchclusters) };
//        int achannels[] = { 0 }; // 1 channel, number 0
//        float atranges[] = { 0, static_cast<float>(m_patchclusters)}; // thermal intensity values range: [0, 1)
//        const float* aranges[] = { atranges };
//        
//        cv::Mat classHist;
//        calcHist(&classes, 1, achannels, noArray(), classHist, 1, ahistSize, aranges, true, false);
//        classes.release();
//        
//        cout << "Class distribution [0-numclasses]: " << classHist << endl;
//        cout << endl;
//    }
//
//}
