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

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::accuracy(cv::Mat actuals, GridMat predictions, cv::Mat& accuracies)
{
    accuracies.create(predictions.crows(), predictions.ccols(), cv::DataType<float>::type);
    
    int nobjects  = cv::sum(actuals == 0).val[0];
    int nsubjects = cv::sum(actuals == 1).val[0];
    
    for (int i = 0; i < predictions.crows(); i++) for (int j = 0; j < predictions.ccols(); j++)
    {
        int objectHits  = 0;
        int subjectHits = 0;
        
        for (int k = 0; k < actuals.rows; k++)
        {
            int actualVal = actuals.at<int>(k,0);
//            int predVal = predictions.at<int>(i,j,k,0);
//            
//            if (actualVal == 0 && predVal == 0) objectHits++;
//            else if (actualVal == 1 && predVal == 1) subjectHits++;
        }
        
//        accuracies.at<float>(i,j) = ( ((float)subjectHits)/nsubjects + ((float)objectHits)/nobjects ) / 2.0;
    }
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

template<class PredictorT>
void ModalityPrediction<PredictorT>::accuracy(cv::Mat actuals, GridMat predictions, cv::Mat& accuracies)
{
    return ModalityPredictionBase<PredictorT>:: accuracy(actuals, predictions);
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
    
    GridMat predictionsTe;
    GridMat loglikelihoodsTe;
    vector<GridPredictor<cv::EM> > predictors;
    
    for (int i = 0; i < m_testK; i++)
    {
        ModalityGridData dataTrFold (m_data, partitions != i);
        ModalityGridData dataTeFold (m_data, partitions == i);
        GridMat descriptorsTrFold (m_descriptors, partitions != i);
        GridMat descriptorsTeFold (m_descriptors, partitions == i);
        
        GridMat selectedParams;
        modelSelection(dataTrFold, descriptorsTrFold,
                       m_nmixtures, m_logthresholds,
                       selectedParams);
        
        GridPredictor<cv::EM> predictor;
        predictor.setData(descriptorsTrFold, dataTrFold.getTags());
        predictor.setParameters(selectedParams);
        predictor.train();
        
        GridMat predictionsTeFold, loglikelihoodsTeFold;
        predictor.predict(descriptorsTeFold, predictionsTeFold, loglikelihoodsTeFold);
        
        predictionsTe.vset(predictionsTeFold, partitions == i);
        loglikelihoodsTe.vset(loglikelihoodsTeFold, partitions == i);
    }
}


void expandParameters(vector<vector<double> > params, cv::Mat& expandedParams)
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

// Given the expanded list of parameters' combinations, and a GridMat
// of 2dim indices from which the first element index the number of the
// row in parameters, return a GridMat of hp-by-wp vectors of parameters.
void selectParameters(cv::Mat parameters, GridMat indices, GridMat& selection)
{
//    selection.create<int>(indices.crows(), indices.ccols(), 1, parameters.cols);
//    for (int i = 0; selection.crows(); i++) for (int j = 0; selection.ccols(); j++)
//    {
//        int idx = indices.at<int>(i,j,0,0);
//        parameters.row(idx).copyTo(selection.at(i,j).row(0));
//    }
}


void ModalityPrediction<cv::EM>::modelSelection(ModalityGridData data, GridMat descriptors,
                                                vector<int> nmixtures, vector<int> loglikelihoods,
                                                GridMat& selection)
{
    // Prepare parameters' combinations
    
    vector<vector<double> > params;
    params.push_back(vector<double>(nmixtures.begin(), nmixtures.end()));
    params.push_back(vector<double>(loglikelihoods.begin(), loglikelihoods.end()));
    
    cv::Mat expandedParams;
    expandParameters(params, expandedParams);
    
    // Partitionate the data in folds
    
    cv::Mat partitions;
    cvpartition(m_data.getTags(), m_modelSelecK, m_seed, partitions);
    
    // Instanciate a hp-by-wp GridMat of accuracies. A cell contains a matrix
    // being the rows the parameters' combinations and columns fold-runs
    GridMat accuracies;
    
    vector<GridPredictor<cv::EM> > predictors;
    for (int k = 0; k < m_modelSelecK; k++)
    {
        GridMat foldAccs;
        for (int m = 0; m < expandedParams.rows; m++)
        {
            // Get fold's data
            ModalityGridData dataTr (data, partitions != k);
            ModalityGridData dataVal (data, partitions == k);
            GridMat descriptorsTr (descriptors, partitions != k);
            GridMat descriptorsVal (descriptors, partitions == k);
            
            // Create predictor and its parametrization
            GridPredictor<cv::EM> predictor;
            predictor.setData(descriptorsTr, dataTr.getTags());
            
            cv::Mat nmixtures (data.hp(), data.wp(), cv::DataType<int>::type);
            cv::Mat loglikes (data.hp(), data.wp(), cv::DataType<int>::type);
            
            nmixtures.setTo(expandedParams.at<double>(m,0));
            loglikes.setTo(expandedParams.at<double>(m,1));
            
            predictor.setNumOfMixtures(nmixtures);
            predictor.setLoglikelihoodThreshold(loglikes);
            
            // Train
            predictor.train();
            
            // Test
            GridMat predictionsVal, loglikelihoodsVal;
            predictor.predict(descriptorsVal, predictionsVal, loglikelihoodsVal);
            
            // Compute an accuracy measure
            cv::Mat accs; // (m_hp * m_wp) accuracies get by params combination in k-th fold
            accuracy(dataVal.getTags(), predictionsVal, accs);
            
            GridMat paramsAccs (accs, data.hp(), data.wp());
            foldAccs.vconcat(paramsAccs);
        }
        accuracies.hconcat(foldAccs);
    }
    
    GridMat foldsMeanAcc, foldsArgmaxAcc;
    accuracies.mean(foldsMeanAcc, 1);
//    foldsMeanAcc.argmax<float>(foldsArgmaxAcc);
//    selectParameters(expandedParams, foldsArgmaxAcc, selection);
}

void ModalityPrediction<cv::EM>::accuracy(cv::Mat actuals, GridMat predictions, cv::Mat& accuracies)
{
    return ModalityPredictionBase<cv::EM>::accuracy(actuals, predictions, accuracies);
}


// Explicit template instanciation (to avoid linking errors)
template class ModalityPredictionBase<cv::EM>;
template class ModalityPrediction<cv::EM>;