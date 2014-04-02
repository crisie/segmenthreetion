//
//  FusionPrediction.h
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#ifndef __segmenthreetion__FusionPrediction__
#define __segmenthreetion__FusionPrediction__

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ModalityGridData.hpp"
#include "GridMat.h"

using namespace std;


/*
 * Simple fusion prediction
 */
template<typename PredictorT>
class SimpleFusionPrediction
{
};

template<>
class SimpleFusionPrediction<cv::EM>
{
public:
    SimpleFusionPrediction();
    
    void setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods);
    void setModalitiesPredictions(vector<GridMat> predictions);
    // TODO: set the loglikelihoods' thresholds
    
    void predict();
    
private:
    vector<GridMat> m_loglikelihoods;
    vector<GridMat> m_predictions;
};


/*
 * Classifier-based prediction
 */

// Template definition NEEDED, but NOT USED. Use the specializations nextly defined
template<typename PredictorT, typename ClassifierT>
class ClassifierFusionPredictionBase
{
};

// <cv::EM,ClassifierT> template parital-instantation
template<typename ClassifierT>
class ClassifierFusionPredictionBase<cv::EM, ClassifierT>
{
public:
    ClassifierFusionPredictionBase();
    
    void setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods);
    void setModalitiesPredictions(vector<GridMat> predictions);
    
private:
    GridMat m_loglikelihoods;
    GridMat m_predictions;
    cv::Mat m_data;
    ClassifierT* m_pClassifier;
    
    void formatData();
};

// SVM template

// Template definition NEEDED, but NOT USED. Use the specializations nextly defined
template<typename PredictorT, typename ClassifierT>
class ClassifierFusionPrediction : public ClassifierFusionPredictionBase<PredictorT, ClassifierT>
{};

// <cv::EM,CvSVM> template instantiation
template<>
class ClassifierFusionPrediction<cv::EM,CvSVM> : public ClassifierFusionPredictionBase<cv::EM,CvSVM>
{
public:
    ClassifierFusionPrediction();
    
};

#endif /* defined(__segmenthreetion__FusionPrediction__) */
