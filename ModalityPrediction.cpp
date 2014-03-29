
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
void ModalityPredictionBase<PredictorT>::setData(ModalityGridData& data)
{
    m_data = data;
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
void ModalityPredictionBase<PredictorT>::expandParameters(vector<vector<double> > params,
                                                          vector<vector<double> >& expandedParams)
{
    variate(params, expandedParams);
    for (int i = 0; i < expandedParams.size(); i++)
    {
        cv::Mat m (expandedParams[i].size(), 1, cv::DataType<double>::type, expandedParams[i].data());
        cout << m << endl;
    }
}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::expandParameters(vector<vector<double> > params,
                                                          int ncells, vector<vector<double> >& gridExpandedParams)
{
    vector<vector<double> > cellExpandedParams;
    variate(params, cellExpandedParams);
    
    vector<int> indices(cellExpandedParams.size());
    
    // Fill with 0,1,2,3,...,n
    for (int i = 0; i < cellExpandedParams.size(); i++)
        indices[i] = i;
        
    vector<vector<int> > listsOfIndices(ncells);
    for (int i = 0; i < ncells; i++)
        listsOfIndices[i] = indices;
    
    vector<vector<int> > expandedIndices;
    variate(listsOfIndices, expandedIndices);
    
//    // debug
//    for (int i = 0; i < expandedIndices.size(); i++)
//    {
//        cv::Mat m (expandedIndices[i].size(), 1, cv::DataType<int>::type, expandedIndices[i].data());
//        cout << m << endl;
//    }
//    //
    
    gridExpandedParams.clear();
    gridExpandedParams.resize(expandedIndices.size());
    
    for (int i = 0; i < expandedIndices.size(); i++)
    {
        for (int j = 0; j < expandedIndices[i].size(); j++)
        {
            vector<double> combination = cellExpandedParams[expandedIndices[i][j]];
            for (int k = 0; k < params.size(); k++)
            {
                gridExpandedParams[i].push_back(combination[k]);
            }
        }
        cv::Mat m (gridExpandedParams[i].size(), 1, cv::DataType<double>::type, gridExpandedParams[i].data());
        cout << m << endl;
    }
}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp,wp,CV_64F));
    
    vector<double> lineParams = expandedParams[idx];
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<double>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
//    for (int k = 0; k < nparams; k++)
//        cout << selectedParams[k] << endl;
}

template<class PredictorT>
void ModalityPredictionBase<PredictorT>::accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
{
    accuracies.create(predictions.crows(), predictions.ccols(), cv::DataType<float>::type);
    
    for (int i = 0; i < predictions.crows(); i++) for (int j = 0; j < predictions.ccols(); j++)
    {
        int nobjects  = cv::sum(actuals.at(i,j) == 0).val[0] / std::numeric_limits<unsigned char>::max();
        int nsubjects = cv::sum(actuals.at(i,j) == 1).val[0] / std::numeric_limits<unsigned char>::max();
        
        int objectHits  = 0;
        int subjectHits = 0;
        
        // label homogeinization
        
        double minVal, maxVal;
        
        cv::minMaxIdx(actuals.at(i,j), &minVal, &maxVal);
        cv::Mat actualsMat = actuals.at(i,j) - minVal;
        
        cv::minMaxIdx(predictions.at(i,j), &minVal, &maxVal);
        cv::Mat predictionsMat = predictions.at(i,j) - minVal;
        
        for (int k = 0; k < actuals.at(i,j).rows; k++)
        {
            int actualVal = actualsMat.at<int>(k,0);
            int predVal = predictionsMat.at<int>(k,0);
            
            if (actualVal == 0 && predVal == 0) objectHits++;
            else if (actualVal == 1 && predVal == 1) subjectHits++;
        }
        
        accuracies.at<float>(i,j) = ( ((float)subjectHits)/nsubjects + ((float)objectHits)/nobjects ) / 2.0;
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
void ModalityPrediction<PredictorT>::setData(ModalityGridData& data)
{
    ModalityPredictionBase<PredictorT>::setData(data);
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
void ModalityPrediction<PredictorT>::expandParameters(vector<vector<double> > params, vector<vector<double> >& expandedParams)
{
    return ModalityPredictionBase<PredictorT>::expandParameters(params, expandedParams);
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::expandParameters(vector<vector<double> > params, int ncells, vector<vector<double> >& gridExpandedParams)
{
    return ModalityPredictionBase<PredictorT>::expandParameters(params, ncells, gridExpandedParams);
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    return ModalityPredictionBase<PredictorT>::selectParameterCombination(expandedParams, hp, wp, nparams, idx, selectedParams);
}

template<class PredictorT>
void ModalityPrediction<PredictorT>::accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
{
    return ModalityPredictionBase<PredictorT>::accuracy(actuals, predictions);
}


//
// ModalityPrediction<cv::EM>
//

ModalityPrediction<cv::EM>::ModalityPrediction()
: ModalityPredictionBase<cv::EM>()
{
    
}

void ModalityPrediction<cv::EM>::setData(ModalityGridData& data)
{
    ModalityPredictionBase<cv::EM>::setData(data);
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
    GridMat descriptors = m_data.getDescriptors();
    GridMat tags = m_data.getValidTags();
    
    GridMat partitions;
    cvpartition(tags, m_testK, m_seed, partitions);
    
    vector<GridPredictor<cv::EM> > predictors;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << "Out-of-sample CV. It: " << k << endl;
        
        GridMat tagsTrFold (tags, partitions, k, true);
        GridMat tagsTeFold (tags, partitions, k);
        GridMat descriptorsTrFold (descriptors, partitions, k, true);
        GridMat descriptorsTeFold (descriptors, partitions, k);
        
        GridMat descriptorsSubjTr (descriptorsTrFold, tagsTrFold, 1);
        
        GridMat selectedParams;
        modelSelection(descriptorsTrFold, tagsTrFold,
                       m_nmixtures, m_logthresholds,
                       selectedParams);
        
        GridPredictor<cv::EM> predictor;
        predictor.setData(descriptorsTrFold);
        predictor.setParameters(selectedParams);
        predictor.train();
        
        GridMat predictionsTeFold, loglikelihoodsTeFold;
        predictor.predict(descriptorsTeFold, predictionsTeFold, loglikelihoodsTeFold);
        
//        predictions.vset(predictionsTeFold, partitions == k);
//        loglikelihoods.vset(loglikelihoodsTeFold, partitions == k);
    }
}

void ModalityPrediction<cv::EM>::modelSelection(GridMat descriptors, GridMat tags,
                                                vector<int> nmixtures, vector<int> loglikelihoods,
                                                GridMat& selection)
{
    // Prepare parameters' combinations
    
    vector<vector<double> > params;
    params.push_back(vector<double>(nmixtures.begin(), nmixtures.end()));
    params.push_back(vector<double>(loglikelihoods.begin(), loglikelihoods.end()));
    
    cout << "Expanding parameters.." << endl;
    vector<vector<double> > expandedParams;
    expandParameters(params, m_data.getHp() * m_data.getWp(), expandedParams);
    //cout << expandedParams << endl;
    
    // Partitionate the data in folds
    
    GridMat partitions;
    cvpartition(tags, m_modelSelecK, m_seed, partitions);
    
    // Instanciate a hp-by-wp GridMat of accuracies. A cell contains a matrix
    // being the rows the parameters' combinations and columns fold-runs
    GridMat accuracies;
    
    vector<GridPredictor<cv::EM> > predictors;
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << "Model selection CV. It: " << k << endl;
        
        // Get fold's data
        GridMat tagsTr (tags, partitions, k, true);
        GridMat tagsVal (tags, partitions, k);
        GridMat descriptorsTr (descriptors, partitions, k, true);
        GridMat descriptorsVal (descriptors, partitions, k);
        
        GridMat descriptorsSubjTr (descriptorsTr, tagsTr, 1); // subjects' training sample
        
        GridMat descriptorsSubjObjVal (descriptorsVal, tagsVal, -1, true);
        GridMat tagsSubjObjVal (tagsVal, tagsVal, -1, true);
        
        GridMat foldAccs; // results
        
        for (int m = 0; m < expandedParams.size(); m++)
        {
            cout << "param. comb.: " << m + 1 << "/" << expandedParams.size() << endl;
            
            // Create predictor and its parametrization
            GridPredictor<cv::EM> predictor;
            predictor.setData(descriptorsSubjTr); // unsupevised method, thus no tags
            
            cv::Mat nmixtures (m_data.getHp(), m_data.getWp(), cv::DataType<int>::type);
            cv::Mat loglikes (m_data.getHp(), m_data.getWp(), cv::DataType<int>::type);
            
            vector<cv::Mat> selectedParams;
            selectParameterCombination(expandedParams, m_data.getHp(), m_data.getWp(), params.size(), m, selectedParams);
            
            nmixtures.setTo(selectedParams[0]); // TODO: fix
            loglikes.setTo(selectedParams[1]);
            
            predictor.setNumOfMixtures(nmixtures);
            predictor.setLoglikelihoodThreshold(loglikes);

            // Train
            predictor.train();
            
            // Test
            GridMat predictionsVal, loglikelihoodsVal;
            predictor.predict(descriptorsSubjObjVal, predictionsVal, loglikelihoodsVal);
            
            // Compute an accuracy measure
            cv::Mat accs; // (m_hp * m_wp) accuracies get by params combination in k-th fold
            accuracy(tagsSubjObjVal, predictionsVal, accs);
            cout << accs << endl;
            
            GridMat paramsAccs (accs, m_data.getHp(), m_data.getWp());
            foldAccs.vconcat(paramsAccs);
        }
        
        accuracies.hconcat(foldAccs);
    }
    
    GridMat foldsMeanAcc, foldsArgmaxAcc;
    accuracies.mean(foldsMeanAcc, 1);
//    foldsMeanAcc.argmax<float>(foldsArgmaxAcc);
//    selectParameters(expandedParams, foldsArgmaxAcc, selection);
}


void ModalityPrediction<cv::EM>::expandParameters(vector<vector<double> > params, vector<vector<double> >& expandedParams)
{
    return ModalityPredictionBase<cv::EM>::expandParameters(params, expandedParams);
}


void ModalityPrediction<cv::EM>::expandParameters(vector<vector<double> > params, int ncells, vector<vector<double> >& gridExpandedParams)
{
    return ModalityPredictionBase<cv::EM>::expandParameters(params, ncells, gridExpandedParams);
}

void ModalityPrediction<cv::EM>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    return ModalityPredictionBase<cv::EM>::selectParameterCombination(expandedParams, hp, wp, nparams, idx, selectedParams);
}

void ModalityPrediction<cv::EM>::accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
{
    return ModalityPredictionBase<cv::EM>::accuracy(actuals, predictions, accuracies);
}


// Explicit template instanciation (to avoid linking errors)
template class ModalityPredictionBase<cv::EM>;
//template class ModalityPrediction<cv::EM>;