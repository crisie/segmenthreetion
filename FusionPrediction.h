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
{};

template<>
class SimpleFusionPrediction<cv::EM>
{
public:
    
    SimpleFusionPrediction();
    
    void setModalitiesLoglikelihoods(vector<GridMat> loglikelihoods);
    void setModalitiesPredictions(vector<GridMat> predictions);
    // TODO: set the loglikelihoods' thresholds
    
    void predict(GridMat& predictions);
    
private:
    
    // Attributes
    
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
    
    void setModelSelection(int k, bool best);
    void setModelValidation(int k, int seed);
    
private:
    
    void formatData();
    
    // Attributes
    
    vector<GridMat> m_loglikelihoods;
    vector<GridMat> m_predictions;
    cv::Mat m_data;
    ClassifierT* m_pClassifier;
    
    int m_testK;
    int m_modelSelecK;
    bool m_selectBest;
    
    int m_seed;
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
    
    void setKernelType(int type);
    
    void setCs(vector<float> cs);
    void setGammas(vector<float> gammas);
    
    void predict(GridMat& predictions);

private:
    
    // Attributes
    
    int m_kernelType; // CvSVM::LINEAR or CvSVM::RBF
    vector<float> m_cs;
    vector<float> m_gammas;
};

#endif /* defined(__segmenthreetion__FusionPrediction__) */
