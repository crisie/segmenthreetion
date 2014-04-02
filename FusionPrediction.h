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
#include "CvExtraTools.h"

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
    
    void setData(vector<GridMat> loglikelihoods, vector<GridMat> predictions);
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
    
    void setData(vector<GridMat> loglikelihoods, vector<GridMat> predictions);
    void setResponses(cv::Mat responses);
    
    void setModelSelection(int k, bool best);
    void setModelValidation(int k, int seed);
    
protected:
    
    void formatData();

    // Attributes
    
    vector<GridMat> m_loglikelihoods;
    vector<GridMat> m_predictions;
    
    cv::Mat m_data; // input data
    cv::Mat m_responses; // output labels
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
    
    template<typename T>
    void modelSelection(cv::Mat data, cv::Mat responses, vector<vector<T> > params, cv::Mat& goodnesses);
    
    void compute(cv::Mat& predictions);

private:
    
    // Attributes
    
    int m_kernelType; // CvSVM::LINEAR or CvSVM::RBF
    vector<float> m_cs;
    vector<float> m_gammas;
};

#endif /* defined(__segmenthreetion__FusionPrediction__) */
