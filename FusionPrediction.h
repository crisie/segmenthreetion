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
#include "em.h"

#include <boost/thread.hpp>

using namespace std;


/*
 * Simple fusion prediction
 */

class SimpleFusionPrediction
{
public:
    
    SimpleFusionPrediction();
    
    void setModalitiesData(vector<ModalityGridData> mgds);
    
    // Cells' preconsensus

    
     // want back some kind of consensued dists to margin in the fusion prediction?
    
    void predict(vector<cv::Mat> allPredictions, vector<cv::Mat> allDistsToMargin,
                 cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin);
    void predict(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin);
    void predict(vector<GridMat> allDistsToMargin, cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin);
    
    cv::Mat getAccuracies();
    
private:
    
    void predict(GridMat distsToMarginGrid, GridMat& fusionPredictions, GridMat& fusionDistsToMargin);
    void computeGridConsensusPredictions(GridMat fusionPredictionsGrid,
                                         GridMat fusionDistsToMarginGrid,
                                         cv::Mat& consensusfusionPredictions,
                                         cv::Mat& consensusfusionDistsToMargin);
    
    // Attributes
    
    vector<ModalityGridData> m_mgds;
    cv::Mat m_tags;
    cv::Mat m_partitions;
    
    cv::Mat m_fusionPredictions;
    cv::Mat m_fusionDistsToMargin;
};


/*
 * Classifier-based prediction
 */

// Template definition NEEDED, but NOT USED. Use the specializations nextly defined
template<typename PredictorT, typename ClassifierT>
class ClassifierFusionPredictionBase
{
    
};

// <cv::EM40,ClassifierT> template parital-instantation
template<typename ClassifierT>
class ClassifierFusionPredictionBase<cv::EM40, ClassifierT>
{
public:
    
    ClassifierFusionPredictionBase();
    
    void setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
    
    void setModelSelection(bool flag);
    void setModelSelectionParameters(int k, int seed, bool bGlobalBest);
    void setValidationParameters(int k);
    void setPartitions(cv::Mat partitions);
    
    void setStackedPrediction(bool flag);
    
    cv::Mat getAccuracies();
    
protected:
    
    void formatData();

    // Attributes
    
    vector<ModalityGridData> m_mgds;
    vector<cv::Mat> m_predictions;
    vector<GridMat> m_distsToMargin;
    
    cv::Mat m_data; // input data
    cv::Mat m_responses; // output labels
    ClassifierT* m_pClassifier;
    cv::Mat m_fusionPredictions;
    
    bool m_bModelSelection;
    int m_testK;
    int m_modelSelecK;
    bool m_bGlobalBest;
    
    int m_seed;
    cv::Mat m_partitions;
    
    bool m_bStackPredictions;
    
    int  m_narrowSearchSteps;
    
    boost::mutex m_mutex;
};

// SVM template

// Template definition NEEDED, but NOT USED. Use the specializations nextly defined
template<typename PredictorT, typename ClassifierT>
class ClassifierFusionPrediction : public ClassifierFusionPredictionBase<PredictorT, ClassifierT>
{};

// <cv::EM40,CvSVM> template instantiation
template<>
class ClassifierFusionPrediction<cv::EM40,CvSVM> : public ClassifierFusionPredictionBase<cv::EM40,CvSVM>
{
public:
    
    ClassifierFusionPrediction();
    
    void setKernelType(int type);
    
    void setCs(vector<float> cs);
    void setGammas(vector<float> gammas);
    
    void modelSelection(cv::Mat data, cv::Mat responses, cv::Mat params, cv::Mat& goodnesses);
    
    void predict(cv::Mat& fusionPredictions);

private:
    void _modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies);
    
    // Attributes
    
    int m_kernelType; // CvSVM::LINEAR or CvSVM::RBF
    vector<float> m_cs;
    vector<float> m_gammas;

    int m_numItersSVM;
};

// <cv::EM40,CvBoost> template instantiation
template<>
class ClassifierFusionPrediction<cv::EM40,CvBoost> : public ClassifierFusionPredictionBase<cv::EM40,CvBoost>
{
public:
    
    ClassifierFusionPrediction();
    
    void setBoostType(int type);
    
    void setNumOfWeaks(vector<float> numOfWeaks);
    void setWeightTrimRate(vector<float> weightTrimRates);
    
    void modelSelection(cv::Mat data, cv::Mat responses, cv::Mat params, cv::Mat& goodnesses);
    
    void predict(cv::Mat& fusionPredictions);
    
private:
    void _modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies);
    
    // Attributes
    
    int m_boostType;
    vector<float> m_numOfWeaks;
    vector<float> m_weightTrimRate;
};

// <cv::EM40,CvANN_MLP> template instantiation
template<>
class ClassifierFusionPrediction<cv::EM40,CvANN_MLP> : public ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>
{
public:
    
    ClassifierFusionPrediction();
    
    cv::Mat encode(cv::Mat vector);
    cv::Mat decode(cv::Mat matrix);

    void setActivationFunctionType(int type);
    void setHiddenLayerSizes(vector<float> hiddenSizes);
    
    //void setBackpropDecayWeightScales(vector<float> dwScales);
    //void setBackpropMomentScales(vector<float> momScales);
    
    void modelSelection(cv::Mat data, cv::Mat responses, cv::Mat params, cv::Mat& goodnesses);
    
    void predict(cv::Mat& fusionPredictions);
    
private:
    void _modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies);
    
    // Attributes
    int m_actFcnType; // activation function type
    
    int m_numOfEpochs;
    int m_numOfRepetitions; // deal with stochasticity introduced by the random weights initialization
    
    vector<float> m_hiddenLayerSizes;
    //vector<float> m_bpDwScales; // decay in weight (dw)
    //vector<float> m_bpMomentScales;
};



#endif /* defined(__segmenthreetion__FusionPrediction__) */
