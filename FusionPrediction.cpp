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
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModalitiesPredictions(vector<GridMat> predictions);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods);
// -----------------------------------------------------------------------------


SimpleFusionPrediction<cv::EM>::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction<cv::EM>::setModalitiesPredictions(vector<GridMat> predictions)
{
    m_predictions = predictions;
}

void SimpleFusionPrediction<cv::EM>::setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods)
{
    m_loglikelihoods = loglikelihoods;
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
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModalitiesPredictions(vector<GridMat> predictions)
{
    m_predictions = predictions;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods)
{
    m_loglikelihoods = loglikelihoods;
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
        GridMat normLoglikes = m_loglikelihoods[i].getNormalizedLoglikelihoods();
        GridMat sparseNormLoglikes;
        normLoglikes.convertToSparse(m_loglikelihoods[i].getValidnesses(), sparseNormLoglikes);
        
        cv::Mat serialMat;
        normLoglikes.hserial(serialMat);
        
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

void ClassifierFusionPrediction<cv::EM,CvSVM>::predict(GridMat &predictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params, expandedParameters;
    params.push_back(m_cs);
    params.push_back(m_gammas);
    
    // create a list of parameters' variations
    expandParameters(params, expandedParameters);
}


