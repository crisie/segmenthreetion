//
//  FusionPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#include "FusionPrediction.h"

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
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::formatData()
{
    // Build here a data structure needed to feed the classifier
    // Better use GridMat functions...

}


//
// ClassifierFusionPrediction class templates' specialization
//


