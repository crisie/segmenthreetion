//
//  FusionPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#include "FusionPrediction.h"
#include "StatTools.h"
#include <boost/assign/std/vector.hpp>

using namespace boost::assign;

// Instantiation of template member functions
// -----------------------------------------------------------------------------

template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setData(vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setModelSelectionParameters(int, bool);
template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setValidationParameters(int, int);
template void ClassifierFusionPredictionBase<cv::EM,CvBoost>::setStackedPrediction(bool flag);

template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setData(vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setModelSelectionParameters(int, bool);
template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setValidationParameters(int, int);
template void ClassifierFusionPredictionBase<cv::EM,CvANN_MLP>::setStackedPrediction(bool flag);

template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setData(vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModelSelectionParameters(int, bool);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setValidationParameters(int, int);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setStackedPrediction(bool flag);
// -----------------------------------------------------------------------------


SimpleFusionPrediction<cv::EM>::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction<cv::EM>::compute(vector<cv::Mat> allPredictions, vector<cv::Mat> allDistsToMargin,
                                             cv::Mat& fusedPredictions, cv::Mat& fusedDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    cv::Mat predictions = allPredictions[0];
    cv::Mat distsToMargin = allDistsToMargin[0];
    
    for (int i = 1; i < allPredictions.size(); i++)
    {
        cv::hconcat(predictions,   allPredictions[i],   predictions);
        cv::hconcat(distsToMargin, allDistsToMargin[i], distsToMargin);
    }
    
    int n = predictions.rows;
    
    fusedPredictions.create(predictions.rows, 1, predictions.type());
    fusedDistsToMargin.create(distsToMargin.rows, 1, distsToMargin.type());
    
    for (int k = 0; k < n; k++)
    {
        int pos = 0;
        int neg = 0;
        float accPosDists = 0;
        float accNegDists = 0;
        for (int m = 0; m < predictions.cols; m++)
        {
            float dist = distsToMargin.at<float>(k,m);
            if (predictions.at<int>(k,m) == 0)
            {
                neg++;
                accNegDists += dist;
            }
            else
            {
                pos++;
                accPosDists += dist;
            }
        }
        
        if (pos < neg)
        {
            fusedPredictions.at<int>(k,0) = 0;
            fusedDistsToMargin.at<float>(k,0) = accNegDists/neg;
        }
        else if (pos > neg)
        {
            fusedPredictions.at<int>(k,0) = 1;
            fusedDistsToMargin.at<float>(k,0) = accPosDists/pos;
        }
        else
        {
            if (accPosDists/pos < abs(accNegDists/neg))
            {
                fusedPredictions.at<int>(k,0) = 0;
                fusedDistsToMargin.at<float>(k,0) = accNegDists/neg;
            }
            else
            {
                fusedPredictions.at<int>(k,0) = 1;
                fusedDistsToMargin.at<float>(k,0) = accPosDists/pos;
            }
                
        }
    }
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    int hp = allPredictions[0].crows();
    int wp = allPredictions[0].ccols();
    for (int m = 1; m < allPredictions.size(); m++)
    {
        assert (allPredictions[m].crows() == hp || allPredictions[m].ccols() == wp);
        assert (allDistsToMargin[m].crows() == hp || allDistsToMargin[m].ccols() == wp);
    }
    
    fusedPredictions.create(hp, wp);
    fusedDistsToMargin.create(hp, wp);
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        vector<cv::Mat> predictions, distsToMargin;
        for (int m = 0; m < allPredictions.size(); m++)
        {
            predictions += allPredictions[m].at(i,j);
            distsToMargin += allDistsToMargin[m].at(i,j);
        }

        compute(predictions, distsToMargin, fusedPredictions.at(i,j), fusedDistsToMargin.at(i,j));
    }
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allDistsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat hConcatDistsToMargin;
    
    for (int i = 0; i < allDistsToMargin.size(); i++)
    {
        hConcatDistsToMargin.hconcat(allDistsToMargin[i]);
    }
    
    compute(hConcatDistsToMargin, fusedPredictions, fusedDistsToMargin);
}

void SimpleFusionPrediction<cv::EM>::compute(GridMat distsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    int hp = distsToMargin.crows();
    int wp = distsToMargin.ccols();
    int n = distsToMargin.at(0,0).rows;
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        assert (n == distsToMargin.at(i,j).rows);
    }
    
    fusedPredictions.create<int>(hp, wp, n, 1);
    fusedDistsToMargin.create<float>(hp, wp, n, 1);
    
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            int pos = 0;
            int neg = 0;
            float accPosDists = 0;
            float accNegDists = 0;
            for (int m = 0; m < distsToMargin.at(i,j).cols; m++)
            {
                float dist = distsToMargin.at<float>(i,j,k,m);
                if (distsToMargin.at<int>(i,j,k,m) < 0)
                {
                    neg++;
                    accNegDists += dist;
                }
                else
                {
                    pos++;
                    accPosDists += dist;
                }
            }
            
            if (pos < neg)
            {
                fusedPredictions.at<int>(i,j,k,0) = 0;
                fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
            }
            else if (pos > neg)
            {
                fusedPredictions.at<int>(i,j,k,0) = 1;
                fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
            }
            else
            {
                float meanNeg = accNegDists/neg;
                float meanPos = accPosDists/pos;
                if (meanPos > abs(meanNeg))
                {
                    fusedPredictions.at<int>(i,j,k,0) = 1;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
                }
                else
                {
                    fusedPredictions.at<int>(i,j,k,0) = 0;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
                }
            }
        }
        
    }
}

//
// ClassifierFusionPredictionBase class
//

template<typename ClassifierT>
ClassifierFusionPredictionBase<cv::EM, ClassifierT>::ClassifierFusionPredictionBase()
: m_pClassifier(new ClassifierT), m_bModelSelection(true), m_bStackPredictions(false), m_narrowSearchSteps(7)
{
    
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<GridMat> distsToMargin)
{
    m_distsToMargin = distsToMargin;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<GridMat> distsToMargin, vector<cv::Mat> predictions)
{
    m_distsToMargin = distsToMargin;
    m_predictions = predictions;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setResponses(cv::Mat responses)
{
    m_responses = responses;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelSelection(bool flag)
{
    m_bModelSelection = flag;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelSelectionParameters(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setValidationParameters(int k, int seed)
{
    m_testK = k;
    m_seed = seed;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::formatData()
{
    m_data.release();
    m_data.create(m_distsToMargin[0].at(0,0).rows, 0, m_distsToMargin[0].at(0,0).type());
    
    // Build here a data structure needed to feed the classifier
    // Better use GridMat functions...
    for (int i = 0; i < m_distsToMargin.size(); i++)
    {
        cv::Mat serialMat;
        m_distsToMargin[i].hserial(serialMat);
        
        if (m_data.cols == 0)
            m_data = serialMat;
        else
            cv::hconcat(m_data, serialMat, m_data);
        
        if (m_bStackPredictions)
        {
            cv::Mat_<float> normPredictions = 2 * m_predictions[i] - 1;
            cv::hconcat(m_data, normPredictions, m_data);
        }
    }
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM,ClassifierT>::setStackedPrediction(bool flag)
{
    m_bStackPredictions = flag;
}


//
// ClassifierFusionPrediction class templates' specialization
//

// SVM

ClassifierFusionPrediction<cv::EM,CvSVM>::ClassifierFusionPrediction()
: m_numItersSVM(10000)
{
    
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setKernelType(int type)
{
    m_kernelType = type;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setCs(vector<float> cs)
{
    m_cs = cs;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setGammas(vector<float> gammas)
{
    m_gammas = gammas;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::compute(cv::Mat& fusionPredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_cs);
    if (m_kernelType == CvSVM::RBF)
        params.push_back(m_gammas);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
    cv::Mat partitions;
    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            cv::Mat narrowExpandedParameters;
            int discretes[] = {0,0};
            narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
            
            std::stringstream coarsess;
            coarsess << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);

            // Narrow search
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    cv::Mat predictions;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodnesses;
        std::stringstream ss;
        ss << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
        cvx::load(ss.str(), goodnesses);
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodnesses.col(goodnesses.cols - 1), &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestC     = goodnesses.row(best.y).at<float>(0,0);
        float bestGamma = goodnesses.row(best.y).at<float>(0,1);
        
        CvSVMParams svmParams (CvSVM::C_SVC, m_kernelType, 0, bestGamma, 0, bestC, 0, 0, 0,
                               cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, 1e-2 ));
        m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(), svmParams);
        
        // Prediction phase
        cv::Mat tePredictions;
        m_pClassifier->predict(teData, tePredictions);
        
        cvx::setMat(tePredictions, predictions, partitions == k);
    }
    cout << endl;
    
    // Mat to GridMat
    fusionPredictions = predictions;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, 0, cv::DataType<float>::type);
    
    //cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        //cout << k << " ";
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat foldAccs (expandedParams.rows, 1, cv::DataType<float>::type); // results
        
        for (int m = 0; m < expandedParams.rows; m++)
        {
            // Training phase
            cv::Mat selectedParams = expandedParams.row(m);
            
            float C = selectedParams.at<float>(0,0);
            
            float gamma = 0;
            if (m_kernelType == CvSVM::RBF)
                gamma = selectedParams.at<float>(0,1); // indeed, gamma not used if not RBF kernel
            
            CvSVMParams params (CvSVM::C_SVC, m_kernelType, 0, gamma, 0, C, 0, 0, 0,
                                cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, 1e-2 ));
            
            m_pClassifier->train(trSbjObjData, trSbjObjResponses, cv::Mat(), cv::Mat(), params);
            
            // Test phase
            cv::Mat valPredictions;
            m_pClassifier->predict(valData, valPredictions);
            
            // Compute an accuracy measure
            foldAccs.at<float>(m,0) = accuracy(valResponses, valPredictions);
        }
        
        if (accuracies.cols == 0)
            accuracies = foldAccs;
        else
            cv::hconcat(accuracies, foldAccs, accuracies); // concatenate along the horizontal direction
    }
    //cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}


// CvBoost

ClassifierFusionPrediction<cv::EM,CvBoost>::ClassifierFusionPrediction()
{
    
}

void ClassifierFusionPrediction<cv::EM,CvBoost>::setBoostType(int type)
{
    m_BoostType = type;
}

void ClassifierFusionPrediction<cv::EM,CvBoost>::setNumOfWeaks(vector<float> numOfWeaks)
{
    m_NumOfWeaks = numOfWeaks;
}

void ClassifierFusionPrediction<cv::EM,CvBoost>::setWeightTrimRate(vector<float> weightTrimRates)
{
    m_WeightTrimRate = weightTrimRates;
}

void ClassifierFusionPrediction<cv::EM,CvBoost>::compute(cv::Mat& fusionPredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_NumOfWeaks);
    if (m_BoostType == CvBoost::GENTLE)
        params.push_back(m_WeightTrimRate);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
    cv::Mat partitions;
    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            cv::Mat narrowExpandedParameters;
            int discretes[] = {1,0};
            narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
            
            std::stringstream coarsess;
            coarsess << "boost_" << m_distsToMargin.size() << "_" << m_BoostType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);
        
            // Narrow search
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "boost_" << m_distsToMargin.size() << "_" << m_BoostType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    cv::Mat predictions;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodnesses;
        std::stringstream ss;
        ss << "boost_" << m_distsToMargin.size() << "_" << m_BoostType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
        cvx::load(ss.str(), goodnesses);
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodnesses.col(goodnesses.cols - 1), &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestNumOfWeaks     = goodnesses.row(best.y).at<float>(0,0);
        float bestWeightTrimRate = goodnesses.row(best.y).at<float>(0,1);
        
        CvBoostParams boostParams (m_BoostType, bestNumOfWeaks, bestWeightTrimRate, 1, 0, NULL);
        
        m_pClassifier->train(validTrData, CV_ROW_SAMPLE, validTrResponses,
                             cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
                             boostParams);
        
        // Test phase
        cv::Mat tePredictions (teResponses.rows, teResponses.cols, teResponses.type());
        for (int d = 0; d < teData.rows; d++)
            tePredictions.at<int>(d,0) = (int) m_pClassifier->predict(teData.row(d));
        
        cvx::setMat(tePredictions, predictions, partitions == k);
    }
    cout << endl;
    
    // Mat to GridMat
    fusionPredictions = predictions;
}

void ClassifierFusionPrediction<cv::EM,CvBoost>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, 0, cv::DataType<float>::type);
    
    //cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        //cout << k << " ";
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat foldAccs (expandedParams.rows, 1, cv::DataType<float>::type); // results
        
        for (int m = 0; m < expandedParams.rows; m++)
        {
            // Training phase
            cv::Mat selectedParams = expandedParams.row(m);
            
            float bestNumOfWeaks = selectedParams.at<float>(0,0);
            float bestWeightTrimRate = selectedParams.at<float>(0,1); // indeed, gamma not used if not RBF kernel
            
            CvBoostParams boostParams (m_BoostType, (int) bestNumOfWeaks, bestWeightTrimRate, 1, 0, NULL);
            
            m_pClassifier->train(trSbjObjData, CV_ROW_SAMPLE, trSbjObjResponses,
                                 cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
                                 boostParams);
            
            // Test phase
            cv::Mat valPredictions (valResponses.rows, valResponses.cols, valResponses.type());
            for (int d = 0; d < valData.rows; d++)
                valPredictions.at<int>(d,0) = (int) m_pClassifier->predict(valData.row(d));
            
            // Compute an accuracy measure
            foldAccs.at<float>(m,0) = accuracy(valResponses, valPredictions);
        }
        
        if (accuracies.cols == 0)
            accuracies = foldAccs;
        else
            cv::hconcat(accuracies, foldAccs, accuracies); // concatenate along the horizontal direction
    }
    //cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}

// CvANN_MLP

ClassifierFusionPrediction<cv::EM,CvANN_MLP>::ClassifierFusionPrediction()
: m_NumOfEpochs(500), m_NumOfRepetitions(5)
{
    
}

void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::setActivationFunctionType(int type)
{
    m_ActFcnType = type;
}

void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::setHiddenLayerSizes(vector<float> hiddenSizes)
{
    m_HiddenLayerSizes = hiddenSizes;
}

//void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::setBackpropDecayWeightScales(vector<float> dwScales)
//{
//    m_bpDwScales = dwScales;
//}
//
//void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::setBackpropMomentScales(vector<float> momScales)
//{
//    m_bpMomentScales = momScales;
//}

void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::compute(cv::Mat& fusionPredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_HiddenLayerSizes);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
    cv::Mat partitions;
    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            cv::Mat narrowExpandedParameters;
            int discretes[] = {1};
            narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
            
            std::stringstream coarsess;
            coarsess << "mlp_" << m_distsToMargin.size() << "_" << m_ActFcnType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);
            
            // Narrow search
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "mlp_" << m_distsToMargin.size() << "_" << m_ActFcnType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    cv::Mat predictions;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (partitions != k) & (partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodnesses;
        std::stringstream ss;
        ss << "mlp_" << m_distsToMargin.size() << "_" << m_ActFcnType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
        cvx::load(ss.str(), goodnesses);
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodnesses.col(goodnesses.cols - 1), &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestHiddenSize    = goodnesses.row(best.y).at<float>(0,0);
        
        cv::Mat layerSizes (3, 1, cv::DataType<int>::type);
        layerSizes.at<int>(0,0) = trData.cols; // as many inputs as feature vectors' num of dimensions
        layerSizes.at<int>(1,0) = bestHiddenSize; // num of hidden neurons experimentally selected
        layerSizes.at<int>(2,0) = 1; // one output neuron
        m_pClassifier->create(layerSizes, m_ActFcnType);
        
        CvANN_MLP_TrainParams mlpParams (cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1, 1e-2), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
        
        validTrResponses.convertTo(validTrResponses, cv::DataType<float>::type);
        m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(),
                             mlpParams);
        
        cv::Mat valPredictions;
        m_pClassifier->predict(valData, valPredictions);
        float prevAcc = accuracy(valResponses, valPredictions);
        
        float bestValAcc = 0;
        
        bool overfitting = false; // wheter performance in validation keeps decreasing
        int counter = 0; // number of iterations the performance in validation is getting worse
        
        bool saturation = false; // performance in training is saturated
        
        cv::Mat tePredictions;
        
        int e;
        for (e = 0; e < m_NumOfEpochs && !overfitting && !saturation; e++)
        {
            int iters = m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(),
                                             mlpParams, CvANN_MLP::UPDATE_WEIGHTS);
            saturation = (iters < 1);
            
            // Test phase
            cv::Mat valPredictions;
            m_pClassifier->predict(valData, valPredictions);
            
            float acc = accuracy(valResponses, valPredictions);
            if (acc > bestValAcc)
            {
                bestValAcc = acc;
                m_pClassifier->predict(teData, tePredictions);
            }
            
            if (acc > prevAcc) counter = 0;
            else overfitting = (++counter == 3); // last e consecutive epochs getting worse
            
            prevAcc = acc;
        }

        cvx::setMat(tePredictions, predictions, partitions == k);
    }
    cout << endl;
    
    // Mat to GridMat
    fusionPredictions = predictions;
}

void ClassifierFusionPrediction<cv::EM,CvANN_MLP>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, 0, cv::DataType<float>::type);
    
    //cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        //cout << k << " ";
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat foldAccs (expandedParams.rows, m_NumOfRepetitions, cv::DataType<float>::type); // results
        
        for (int m = 0; m < expandedParams.rows; m++)
        {
            // Training phase
            cv::Mat selectedParams = expandedParams.row(m);
            
            float hiddenSize     = selectedParams.at<float>(0,0);
            
            cv::Mat layerSizes (3, 1, cv::DataType<int>::type);
            layerSizes.at<int>(0,0) = trData.cols; // as many inputs as feature vectors' num of dimensions
            layerSizes.at<int>(1,0) = hiddenSize; // num of hidden neurons experimentally selected
            layerSizes.at<int>(2,0) = 1; // one output neuron
            
            for (int r = 0; r < m_NumOfRepetitions; r++)
            {
                m_pClassifier->create(layerSizes, m_ActFcnType);
                
                CvANN_MLP_TrainParams mlpParams (cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1, 1e-2), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
                
                trSbjObjResponses.convertTo(trSbjObjResponses, cv::DataType<float>::type);
                m_pClassifier->train(trSbjObjData, trSbjObjResponses, cv::Mat(), cv::Mat(),
                                     mlpParams);
                
                cv::Mat valPredictions;
                m_pClassifier->predict(valData, valPredictions);
                float prevAcc, bestAcc;
                bestAcc = prevAcc = accuracy(valResponses, valPredictions);
                
                bool overfitting = false; // wheter performance in validation keeps decreasing
                int counter = 0; // number of iterations the performance in validation is getting worse
                
                bool saturation = false; // performance in training is saturated
                
                int e;
                for (e = 0; e < m_NumOfEpochs && !overfitting && !saturation; e++)
                {
                    int iters = m_pClassifier->train(trSbjObjData, trSbjObjResponses, cv::Mat(), cv::Mat(),
                                                     mlpParams, CvANN_MLP::UPDATE_WEIGHTS);
                    saturation = (iters < 1);
                    
                    // Test phase
                    cv::Mat valPredictions;
                    m_pClassifier->predict(valData, valPredictions);
                    
                    float acc = accuracy(valResponses, valPredictions);
                    if (acc > bestAcc) bestAcc = acc;
                    
                    if (acc > prevAcc) counter = 0;
                    else overfitting = (++counter == 3);
                    
                    prevAcc = acc;
                }
                
                // Compute an accuracy measure
                foldAccs.at<float>(m,r) = bestAcc;
            }
        }

        cv::Mat tmp;
        cvx::hmean(foldAccs, tmp);
        
        if (accuracies.cols == 0)
            accuracies = tmp;
        else
            cv::hconcat(accuracies, tmp, accuracies); // concatenate along the horizontal direction
    }
    //cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}