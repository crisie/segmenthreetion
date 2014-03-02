//
//  ModalityPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#include "ModalityPrediction.h"
#include "GridPredictor.h"
#include "StatTools.h"

//
// ModalityPredictionBase
//

template<class PredictorT>
ModalityPredictionBase<PredictorT>::ModalityPredictionBase()
{

}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::setData(ModalityGridData data, GridMat descriptors)
{
    m_data = data;
    m_descriptors = descriptors;
}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::setModelSelection(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::setModelValidation(int k, int seed)
{
    m_testK = k;
    m_seed = seed;
}


//
// ModalityPrediction<PredictorT>
//

template<class PredictorT>
ModalityPrediction<PredictorT>::ModalityPrediction()
        : ModalityPredictionBase<PredictorT>()
{
    
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::setData(ModalityGridData data, GridMat descriptors)
{
    ModalityPredictionBase<PredictorT>::setData(data, descriptors);
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::setModelSelection(int k, bool best)
{
    ModalityPredictionBase<PredictorT>::setModelSelection(k, best);
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::setModelValidation(int k, int seed)
{
    ModalityPredictionBase<PredictorT>::setModelValidation(k, seed);
}

//
// ModalityPrediction<cv::EM>
//

ModalityPrediction<cv::EM>::ModalityPrediction()
: ModalityPredictionBase<cv::EM>()
{
    
}

void ModalityPrediction<cv::EM>::setData(ModalityGridData data, GridMat descriptors)
{
    ModalityPredictionBase<cv::EM>::setData(data, descriptors);
}

void ModalityPrediction<cv::EM>::setModelSelection(int k, bool best)
{
    ModalityPredictionBase<cv::EM>::setModelSelection(k, best);
}

void ModalityPrediction<cv::EM>::setModelValidation(int k, int seed)
{
    ModalityPredictionBase<cv::EM>::setModelValidation(k, seed);
}

void ModalityPrediction<cv::EM>::setNumOfMixtures(int m)
{
    m_nmixtures.clear();
    m_nmixtures.push_back(m);
}

void ModalityPrediction<cv::EM>::setNumOfMixtures(vector<int> m)
{
    m_nmixtures = m;
}

void ModalityPrediction<cv::EM>::setLoglikelihoodThresholds(int t)
{
    m_logthresholds.clear();
    m_logthresholds.push_back(t);
}

void ModalityPrediction<cv::EM>::setLoglikelihoodThresholds(vector<int> t)
{
    m_logthresholds = t;
}

void ModalityPrediction<cv::EM>::predict(GridMat& predictions, GridMat& loglikelihoods)
{
    cv::Mat partitions;
    cvpartition(m_data.getTags(), m_testK, m_seed, partitions);
    
    vector<GridPredictor<cv::EM> > predictors;
    for (int i = 0; i < m_testK; i++)
    {
        ModalityGridData dataTr (m_data, partitions != i);
        ModalityGridData dataTe (m_data, partitions == i);
        GridMat descriptorsTr (m_descriptors, partitions != i);
        GridMat descriptorsTe (m_descriptors, partitions == i);
        
        int nmixtures, logthreshold;
        modelSelection(dataTr, m_nmixtures, m_logthresholds, &nmixtures, &logthreshold);
        
        GridPredictor<cv::EM> predictor;
        predictor.setData(descriptorsTr, dataTr.getTags());
        predictor.setNumOfMixtures(nmixtures);
        predictor.setLoglikelihoodThreshold(logthreshold);
        predictor.train();
        
        GridMat predictionsTe, loglikelihoodsTe;
        predictor.predict(descriptorsTe, predictionsTe, loglikelihoodsTe);
        
        // TODO
        // ... merge the results of the test folds ...
        // setResults(predictionsTe, partitions, i, predictions);
        // setResults(loglikelihoodsTe, partitions, i, loglikelihoods);
    }
}

void expand(vector<vector<double> > params, cv::Mat& expandedParams)
{
    expandedParams.release();
    int combinations = 1;
    for (int i = 0; i < params.size(); i++)
        combinations *= params[i].size();
    expandedParams.create(combinations, params.size(), cv::DataType<double>::type);
    
    for (int i = 0; i < params.size(); i++)
    {
        int nextcombinations = 1;
        for (int j = i + 1; j < params.size(); j++)
            nextcombinations *= params[j].size();
        
        for (int j = 0; j < params[i].size(); j++)
        {
            for (int k = 0; k < nextcombinations; k++)
            {
                expandedParams.at<double>(j * nextcombinations + k, i) = params[i][j];
            }
        }
    }
}

void ModalityPrediction<cv::EM>::modelSelection(ModalityGridData data, vector<int> nmixtures, vector<int> loglikelihoods, int* nmixturesSelected, int* loglikelihoodSelected)
{
    // Prepare parameters' combinations
    
    vector<vector<double> > params;
    params.push_back(vector<double>(nmixtures.begin(), nmixtures.end()));
    params.push_back(vector<double>(loglikelihoods.begin(), loglikelihoods.end()));
    
    cv::Mat expandedParams;
    expand(params, expandedParams);
    
    // Partitionate the data in folds
    
    cv::Mat partitions;
    cvpartition(m_data.getTags(), m_modelSelecK, m_seed, partitions);
    
    vector<GridPredictor<cv::EM> > predictors;
    for (int i = 0; i < m_modelSelecK; i++)
    {
        for (int m = 0; m < expandedParams.rows; m++)
        {
            ModalityGridData dataTr (m_data, partitions != i);
            ModalityGridData dataVal (m_data, partitions == i);
            GridMat descriptorsTr (m_descriptors, partitions != i);
            GridMat descriptorsVal (m_descriptors, partitions == i);
            
            GridPredictor<cv::EM> predictor;
            predictor.setData(descriptorsTr, dataTr.getTags());
            predictor.setNumOfMixtures(expandedParams.at<double>(m,0));
            predictor.setLoglikelihoodThreshold(expandedParams.at<double>(m,1));
            predictor.train();
            
            GridMat predictionsVal, loglikelihoodsVal;
            predictor.predict(descriptorsVal, predictionsVal, loglikelihoodsVal);
        }
    }
}


// Explicit template instanciation (to avoid linking errors)
template class ModalityPredictionBase<cv::EM>;
template class ModalityPrediction<cv::EM>;