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
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setData(vector<GridMat> loglikelihoods, vector<GridMat> predictions);

template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<int>(cv::Mat data, cv::Mat responses, vector<vector<int> > params, cv::Mat& goodnesses);
template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<float>(cv::Mat data, cv::Mat responses, vector<vector<float> > params, cv::Mat& goodnesses);
template void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection<double>(cv::Mat data, cv::Mat responses, vector<vector<double> > params, cv::Mat& goodnesses);
// -----------------------------------------------------------------------------


SimpleFusionPrediction<cv::EM>::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction<cv::EM>::setData(vector<GridMat> loglikelihoods, vector<GridMat> predictions)
{
    m_loglikelihoods = loglikelihoods;
    m_predictions = predictions;
}

void SimpleFusionPrediction<cv::EM>::predict(GridMat& predictions)
{
    // TODO: all the stuff
    // ...
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
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<GridMat> loglikelihoods, vector<GridMat> predictions)
{
    m_loglikelihoods = loglikelihoods;
    m_predictions = predictions;
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

void ClassifierFusionPrediction<cv::EM,CvSVM>::compute(cv::Mat& predictions)
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
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, partitions != k);
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(m_responses, partitions != k);
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        
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
        m_pClassifier->train(trData, trResponses, cv::Mat(), cv::Mat(), params);
        
        // Prediction phase
        cv::Mat tePredictions;
        m_pClassifier->predict(teData, tePredictions);
        
        cvx::copyMat(tePredictions, predictions, partitions == k);
    }
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