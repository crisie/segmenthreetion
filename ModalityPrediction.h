//
//  ModalityPrediction.h
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#ifndef __segmenthreetion__ModalityPrediction__
#define __segmenthreetion__ModalityPrediction__

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "GridMat.h"
#include "ModalityGridData.hpp"

using namespace std;

template<typename Prediction>
class ModalityPredictionBase
{
public:
    ModalityPredictionBase();
    
    void setData(ModalityGridData& data);
    
    void setModelSelection(int k, bool best);
    void setModelValidation(int k, int seed);
    
    template<typename T>
    void expandParameters(vector<vector<T> > params, vector<vector<T> >& expandedParams);
    template<typename T>
    void expandParameters(vector<vector<T> > params, int ncells, vector<vector<T> >& expandedParams);
    
    template<typename T>
    void selectParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams,
                                    int idx, vector<cv::Mat>& selectedParams);
    
    template<typename T>
    void selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams,
                                        GridMat goodnesses, vector<cv::Mat>& selectedParams);
    
    void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies);
    
protected:
    ModalityGridData m_data;
    int m_hp, m_wp;
    
    int m_modelSelecK; // number of folds in inner cross-validation to perform model selection
    bool m_selectBest; // in model selection
    
    int m_testK; // number of folds in outer cross-validation to obtain the test results
    int m_seed;
};


template<typename Prediction>
class ModalityPrediction : public ModalityPredictionBase<Prediction>
{
    ModalityPrediction();// : ModalityPredictionBase<Prediction>() {}

//    void setModelSelection(int k = 3, bool best = 0);
//    void setModelValidation(int k = 10, int seed = 74);
//    
//    void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies);
};


template<>
class ModalityPrediction<cv::EM> : public ModalityPredictionBase<cv::EM>
{
public:
    ModalityPrediction(); // : ModalityPredictionBase<cv::EM>() {}
    
//    void setModelSelection(int k = 3, bool best = 0);
//    void setModelValidation(int k = 10, int seed = 74);
    
    void setNumOfMixtures(int m);
    void setNumOfMixtures(vector<int> m);
    
    void setLoglikelihoodThresholds(int t);
    void setLoglikelihoodThresholds(vector<int> t);
    
    template<typename T>
    void modelSelection(GridMat descriptors, GridMat tags,
                        vector<vector<T> > params,
                        GridMat& goodnesses);
    
    void compute(GridMat& predictions, GridMat& loglikelihoods, bool normalizedLoglikelihoods = true);
    
//    void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies);
    
private:
    vector<int> m_nmixtures;
    vector<int> m_logthresholds;
};


template<>
class ModalityPrediction<CvSVM> : public ModalityPredictionBase<CvSVM>
{
public:
    ModalityPrediction(); // : ModalityPredictionBase<cv::EM>() {}
    
//    void setModelSelection(int k = 3, bool best = 0);
//    void setModelValidation(int k = 10, int seed = 74);
    
    void setSvmType(int type);
    void setKernelType(int type);
    
    void setC(float c);
    void setCs(vector<float> c);
    
    void setGamma(float gamma);
    void setGammas(vector<float> gammas);
    
    template<typename T>
    void modelSelection(GridMat descriptors, GridMat tags,
                        vector<vector<T> > params,
                        GridMat& goodnesses);
    
    void compute(GridMat& predictions);
    
//    void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies);
    
private:
    vector<float> m_cs;
    vector<float> m_gammas;
};


#endif /* defined(__segmenthreetion__ModalityPrediction__) */
