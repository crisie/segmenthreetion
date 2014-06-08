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

#include "em.h"
#include "GridMat.h"
#include "ModalityGridData.hpp"

#include <boost/thread.hpp>

using namespace std;

template<typename Prediction>
class ModalityPredictionBase
{
public:
    ModalityPredictionBase();
    
    void setData(ModalityGridData& data);
    
    void setModelSelection(bool flag);
    void setModelSelectionParameters(int k, bool bGlobalBest = false);
    
    void setValidationParameters(int k);
    
    void setDimensionalityReduction(float variance);
    
    void setTrainMirrored(bool flag);
    
    void computeGridConsensusPredictions(cv::Mat& consensusPredictions, cv::Mat& consensusDistsToMargin);

    void getAccuracy(cv::Mat predictions, cv::Mat& accuracies);
    void getAccuracy(GridMat predictions, GridMat& accuracies);
    
    void setPredictions(GridMat predictionsGrid);
    void setDistsToMargin(GridMat distsToMarginGrid);
    
protected:
    
    // Attributes
    
    ModalityGridData m_data;
    int m_hp, m_wp;
    
    int m_seed;
    
    int m_testK; // number of folds in outer cross-validation to obtain the test results
    
    bool m_bModelSelection; // wheter to perform the selection or re-use an old one (in disk)
                            // files would be named: goodnesses_1.yml, ..., goodnesses_N.yml
                            // where N is equal to m_testK
    int m_modelSelecK; // number of folds in inner cross-validation to perform model selection
    bool m_bGlobalBest; // in model selection
    
    bool m_bDimReduction;
    float m_variance;
    
    bool m_bTrainMirrored;
    
    int m_narrowSearchSteps;
    
    GridMat m_PredictionsGrid;
    GridMat m_DistsToMarginGrid;
    
    boost::mutex m_mutex;
};


template<typename Prediction>
class ModalityPrediction : public ModalityPredictionBase<Prediction>
{
    ModalityPrediction();// : ModalityPredictionBase<Prediction>() {}
};


template<>
class ModalityPrediction<cv::EM40> : public ModalityPredictionBase<cv::EM40>
{
public:
    ModalityPrediction();
    
    void setNumOfMixtures(int m);
    void setNumOfMixtures(vector<int> m);
    
    void setEpsilons(float eps);
    void setEpsilons(vector<float> eps);
    
    void setLoglikelihoodThresholds(float t);
    void setLoglikelihoodThresholds(vector<float> t);
    
    template<typename T>
    void modelSelection(GridMat descriptors, GridMat tags,
                        vector<vector<T> > params,
                        GridMat& goodnesses);
    
    void predict(GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin); // this
    
    void computeLoglikelihoodsDistribution(int nbins, double min, double max, cv::Mat& sbjDistribution, cv::Mat& objDistribution);
    
    template<typename T>
    void _modelSelection(GridMat descriptorsSbjTr, GridMat descriptorsSbjObjVal, GridMat tagsSbjObjVal,
                         int k, vector<vector<T> > params, GridMat& accs);
    
private:
    
    // Attributes
    
    vector<int> m_nmixtures;
    vector<float> m_epsilons;
    vector<float> m_logthresholds;
    
    GridMat m_LoglikelihoodsGrid;

};

template<>
class ModalityPrediction<cv::Mat> : public ModalityPredictionBase<cv::Mat>
{
public:
    ModalityPrediction();
    
    void setPositiveClassificationRatios(float m);
    void setPositiveClassificationRatios(vector<float> m);
    
    void setScoreThresholds(float t);
    void setScoreThresholds(vector<float> t);
    
    template<typename T>
    void modelSelection(cv::Mat gridsIndices, cv::Mat tags,
                        unsigned int i, unsigned int j,
                        cv::Mat params,
                        cv::Mat& goodness);
    
    void predict(GridMat& predictions, GridMat& ramananScores, GridMat& distsToMargin); // this
    
private:
    
    // Attributes
    
    vector<float> m_ratios;
    vector<float> m_scores;
    
    GridMat m_RamananScoresGrid;
};


#endif /* defined(__segmenthreetion__ModalityPrediction__) */
