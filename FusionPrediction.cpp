//
//  FusionPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#include "FusionPrediction.h"
#include "StatTools.h"

// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setData(vector<ModalityGridData>&, vector<GridMat> predictions, vector<GridMat> loglikelihoods, vector<GridMat> distsToMargin);

template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<int>(cv::Mat data, cv::Mat responses, vector<vector<int> > params, cv::Mat& goodnesses);
template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<float>(cv::Mat data, cv::Mat responses, vector<vector<float> > params, cv::Mat& goodnesses);
template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<double>(cv::Mat data, cv::Mat responses, vector<vector<double> > params, cv::Mat& goodnesses);

template void ClassifierFusionPredictionBase<cv::EM, CvSVM>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM, CvSVM>::setModelSelection(int, bool);
template void ClassifierFusionPredictionBase<cv::EM, CvSVM>::setModelValidation(int, int);
// -----------------------------------------------------------------------------


SimpleFusionPrediction<cv::EM>::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction<cv::EM>::setData(vector<ModalityGridData>& mgds)
{
    m_mgds = mgds;
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, GridMat& fusedPredictions)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat predictions (allPredictions[0]);
    GridMat distsToMargin (allDistsToMargin[0]);
    
    for (int i = 1; i < m_mgds.size(); i++)
    {
        predictions.hconcat(allPredictions[i]);
        distsToMargin.hconcat(allDistsToMargin[i]);
    }
    
    int elements = predictions.at(0,0).rows;
    for (int i = 0; i < predictions.crows(); i++) for (int j = 1; j < predictions.ccols(); j++)
    {
        assert (elements == predictions.at(i,j).rows);
    }
    
    GridMat negMask (predictions == 0);
    GridMat posMask (predictions == 1);
    
    GridMat hAccNegMask, hAccPosMask;
    negMask.sum(hAccNegMask, 1);
    posMask.sum(hAccPosMask, 1);
    
    GridMat ones (predictions.crows(), predictions.ccols(), elements, 1, 1);

    int majorityVal = ceil(allPredictions.size()/2);
    ones.copyTo(fusedPredictions, hAccPosMask > majorityVal);
    
    GridMat drawsMask = (hAccNegMask <= majorityVal) & (hAccPosMask <= majorityVal);
    GridMat drawDistsToMargin (distsToMargin, drawsMask);
    GridMat drawFusedPredictions;
    
    compute(drawDistsToMargin, drawFusedPredictions);
    
    drawFusedPredictions.set(fusedPredictions, drawsMask);
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allDistsToMargin, GridMat& fusedPredictions)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat hConcatDistsToMargin;
    
    for (int i = 0; i < m_mgds.size(); i++)
    {
        hConcatDistsToMargin.hconcat(allDistsToMargin[i]);
    }

    compute(hConcatDistsToMargin, fusedPredictions);
}

void SimpleFusionPrediction<cv::EM>::compute(GridMat distsToMargin, GridMat& fusedPredictions)
{
    GridMat negMask (distsToMargin < 0);
    GridMat posMask (distsToMargin > 0);
    
    GridMat negDistsToMargin, posDistsToMargin;
    distsToMargin.copyTo(negDistsToMargin, negMask);
    distsToMargin.copyTo(posDistsToMargin, posMask);
    
    GridMat hAccNegMsk, hAccPosMsk, hAccNegDistsToMargin, hAccPosDistsToMargin;
    negMask.sum(hAccNegMsk, 1);
    posMask.sum(hAccPosMsk, 1);
    negDistsToMargin.sum(hAccNegDistsToMargin, 1); // 1 indicates in the horizontal direction
    posDistsToMargin.sum(hAccPosDistsToMargin, 1);
    
    GridMat hMeanNegDistsToMargin, hMeanPosDistsToMargin;
    hMeanNegDistsToMargin = hAccNegDistsToMargin / hAccNegMsk;
    hMeanPosDistsToMargin = hAccPosDistsToMargin / hAccPosMsk;
    
    fusedPredictions = (hMeanPosDistsToMargin > hMeanNegDistsToMargin.abs()) / 255;
}

//
// ClassifierFusionPredictionBase class
//

template<typename ClassifierT>
ClassifierFusionPredictionBase<cv::EM, ClassifierT>::ClassifierFusionPredictionBase()
: m_pClassifier(new ClassifierT)
{
    
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<ModalityGridData>& mgds, vector<GridMat> predictions, vector<GridMat> loglikelihoods, vector<GridMat> distsToMargin)
{
    m_mgds = mgds;
    m_predictions = predictions;
    m_loglikelihoods = loglikelihoods;
    m_distsToMargin = distsToMargin;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setResponses(cv::Mat responses)
{
    m_responses = responses;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelSelection(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelValidation(int k, int seed)
{
    m_testK = k;
    m_seed = seed;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::formatData()
{
    m_data.release();
    m_data.create(m_loglikelihoods[0].at(0,0).rows, 0, m_loglikelihoods[0].at(0,0).type());
    
    // Build here a data structure needed to feed the classifier
    // Better use GridMat functions...
    for (int i = 0; i < m_loglikelihoods.size(); i++)
    {
        cv::Mat serialMat;
        m_loglikelihoods[i].hserial(serialMat);
        
        cv::hconcat(m_data, serialMat, m_data);
    }
}


//
// ClassifierFusionPrediction class templates' specialization
//

ClassifierFusionPrediction<cv::EM,CvSVM>::ClassifierFusionPrediction()
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

void ClassifierFusionPrediction<cv::EM,CvSVM>::compute(GridMat& gpredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params, expandedParameters;
    params.push_back(m_cs);
    if (m_kernelType == CvSVM::RBF)
        params.push_back(m_gammas);
    
    // create a list of parameters' variations
    expandParameters(params, expandedParameters);
    
    cv::Mat partitions;
    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    cout << "Model selection CVs [" << m_testK << "]: " << endl;
    
    vector<cv::Mat> goodnesses(m_testK); // for instance: accuracies
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, partitions != k);
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(m_responses, partitions != k);
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        
        modelSelection(trData, trResponses, expandedParameters, goodnesses[k]);
        
        std::stringstream ss;
        ss << "svm_goodnesses_" << k << ".yml" << endl;
        cvx::save(ss.str(), goodnesses[k]);
    }
    cout << endl;
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    cv::Mat predictions;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, partitions != k);
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(m_responses, partitions != k);
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodnesses;
        std::stringstream ss;
        ss << "svm_goodnesses_" << k << ".yml" << endl;
        cvx::load(ss.str(), goodnesses);
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point min, max;
        cv::minMaxLoc(goodnesses, &minVal, &maxVal, &min, &max);
        
        // Training phase
        vector<float> bestParams = expandedParameters[max.x];
        float bestC = bestParams[0];
        float bestGamma = bestParams[1];
        
        CvSVMParams params (CvSVM::C_SVC, m_kernelType, 0, bestGamma, 0, bestC, 0, 0, 0,
                            cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON ));
        m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(), params);
        
        // Prediction phase
        cv::Mat tePredictions;
        m_pClassifier->predict(teData, tePredictions);
        
        cvx::setMat(tePredictions, predictions, partitions == k);
    }
    
    // Mat to GridMat
    gpredictions.setTo(predictions);
}

template<typename T>
void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection(cv::Mat data, cv::Mat responses, vector<vector<T> > expandedParams, cv::Mat &goodnesses)
{
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.size(), 0, cv::DataType<float>::type);;
    
    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k << " ";
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat foldAccs (expandedParams.size(), 1, cv::DataType<float>::type); // results
        
        for (int m = 0; m < expandedParams.size(); m++)
        {
            // Training phase
            vector<T> selectedParams = expandedParams[m];
            T C = selectedParams[0];
            T gamma = selectedParams[1];
            
            CvSVMParams params (CvSVM::C_SVC, m_kernelType, 0, gamma, 0, C, 0, 0, 0,
                                cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON ));
            m_pClassifier->train(trData, trResponses, cv::Mat(), cv::Mat(), params);
            
            // Test phase
            cv::Mat valPredictions;
            m_pClassifier->train(valData, valPredictions);
            
            // Compute an accuracy measure
            foldAccs.at<float>(m,0) = accuracy(valResponses, valPredictions);
        }
        
        accuracies.push_back(foldAccs); // concatenate along the horizontal direction
    }
    cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}