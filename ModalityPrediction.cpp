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


// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void ModalityPrediction<cv::EM>::modelSelection<int>(GridMat descriptors, GridMat tags, vector<vector<int> > params, GridMat& goodnesses);
template void ModalityPrediction<cv::EM>::modelSelection<double>(GridMat descriptors, GridMat tags, vector<vector<double> > params, GridMat& goodnesses);

template void ModalityPredictionBase<cv::EM>::expandParameters(vector<vector<int> > params, vector<vector<int> >& expandedParams);
template void ModalityPredictionBase<cv::EM>::expandParameters(vector<vector<double> > params, vector<vector<double> >& expandedParams);

template void ModalityPredictionBase<cv::EM>::expandParameters(vector<vector<int> > params, int ncells, vector<vector<int> >& expandedParams);
template void ModalityPredictionBase<cv::EM>::expandParameters(vector<vector<double> > params, int ncells, vector<vector<double> >& expandedParams);

template void ModalityPredictionBase<cv::EM>::selectParameterCombination(vector<vector<int> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);
template void ModalityPredictionBase<cv::EM>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);

template void ModalityPredictionBase<cv::EM>::selectBestParameterCombination(vector<vector<int> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
template void ModalityPredictionBase<cv::EM>::selectBestParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
// -----------------------------------------------------------------------------


//
// ModalityPredictionBase
//

template<typename PredictorT>
ModalityPredictionBase<PredictorT>::ModalityPredictionBase()
{

}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setData(ModalityGridData &data)
{
    m_data = data;
    m_hp = data.getHp();
    m_wp = data.getWp();
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setModelSelection(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setModelValidation(int k, int seed)
{
    m_testK = k;
    m_seed = seed;
}

template<typename PredictorT>
template<typename T>
void ModalityPredictionBase<PredictorT>::expandParameters(vector<vector<T> > params,
                                                          vector<vector<T> >& expandedParams)
{
    variate(params, expandedParams);
//    // debug
//    for (int i = 0; i < expandedParams.size(); i++)
//    {
//        cv::Mat m (expandedParams[i].size(), 1, cv::DataType<double>::type, expandedParams[i].data());
//        cout << m << endl;
//    }
}

template<typename PredictorT>
template<typename T>
void ModalityPredictionBase<PredictorT>::expandParameters(vector<vector<T> > params,
                                                          int ncells, vector<vector<T> >& gridExpandedParams)
{
    vector<vector<T> > cellExpandedParams;
    variate(params, cellExpandedParams);
    
    // Create and expand a list of indices, used to index the cellExpandedParams
    
    vector<int> indices(cellExpandedParams.size());
    
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
    
    // Create the grid's combinations' list of parameters
    
    gridExpandedParams.clear();
    gridExpandedParams.resize(expandedIndices.size());
    
    for (int i = 0; i < expandedIndices.size(); i++)
    {
        for (int j = 0; j < expandedIndices[i].size(); j++)
        {
            vector<T> combination = cellExpandedParams[expandedIndices[i][j]];
            for (int k = 0; k < params.size(); k++)
            {
                gridExpandedParams[i].push_back(combination[k]);
            }
        }
//        // debug
//        cv::Mat m (gridExpandedParams[i].size(), 1, cv::DataType<T>::type, gridExpandedParams[i].data());
//        cout << m << endl;
    }
}

template<typename PredictorT>
template<typename T>
void ModalityPredictionBase<PredictorT>::selectParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
    
    vector<T> lineParams = expandedParams[idx];
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
    //    for (int k = 0; k < nparams; k++)
    //        cout << selectedParams[k] << endl;
}


template<typename PredictorT>
template<typename T>
void ModalityPredictionBase<PredictorT>::selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
    
    GridMat gargmax;
    goodnesses.argmax<T>(gargmax);
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int rowIdx = gargmax.at<T>(i,j,0,0); // maxrow index
        
        vector<T> lineParams = expandedParams[rowIdx];
        
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
    //    for (int k = 0; k < nparams; k++)
    //        cout << selectedParams[k] << endl;
}


template<typename PredictorT>
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

template<typename PredictorT>
ModalityPrediction<PredictorT>::ModalityPrediction()
        : ModalityPredictionBase<PredictorT>()
{
    
}

template<typename PredictorT>
void ModalityPrediction<PredictorT>::setModelSelection(int k, bool best)
{
    ModalityPredictionBase<PredictorT>::setModelSelection(k, best);
}

template<typename PredictorT>
void ModalityPrediction<PredictorT>::setModelValidation(int k, int seed)
{
    ModalityPredictionBase<PredictorT>::setModelValidation(k, seed);
}

//template<typename PredictorT>
//void ModalityPrediction<PredictorT>::expandParameters(vector<vector<double> > params, vector<vector<double> >& expandedParams)
//{
//    return ModalityPredictionBase<PredictorT>::expandParameters(params, expandedParams);
//}
//
//template<typename PredictorT>
//void ModalityPrediction<PredictorT>::expandParameters(vector<vector<double> > params, int ncells, vector<vector<double> >& gridExpandedParams)
//{
//    return ModalityPredictionBase<PredictorT>::expandParameters(params, ncells, gridExpandedParams);
//}
//
//template<typename PredictorT>
//void ModalityPrediction<PredictorT>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
//{
//    return ModalityPredictionBase<PredictorT>::selectParameterCombination(expandedParams, hp, wp, nparams, idx, selectedParams);
//}

template<typename PredictorT>
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
    GridMat tags = m_data.getValidTags(); // as many as descriptors
    
    GridMat partitions;
    cvpartition(tags, m_testK, m_seed, partitions);
    
    vector<GridPredictor<cv::EM> > predictors;
    
    vector<vector<int> > params, gridExpandedParameters;
    
    params.push_back(vector<int>(m_nmixtures.begin(), m_nmixtures.end()));
    params.push_back(vector<int>(m_logthresholds.begin(), m_logthresholds.end()));
    // create a list of parameters' variations
    expandParameters(params, m_hp * m_wp, gridExpandedParameters);
    
//    cout << "Model selection CVs [" << m_testK << "]: " << endl;
//    
//    vector<GridMat> goodnesses(m_testK); // for instance: accuracies
//    
//    for (int k = 0; k < m_testK; k++)
//    {
//        cout << k << " " << endl;
//        
//        GridMat tagsTrFold (tags, partitions, k, true);
//        GridMat tagsTeFold (tags, partitions, k);
//        GridMat descriptorsTrFold (descriptors, partitions, k, true);
//        GridMat descriptorsTeFold (descriptors, partitions, k);
//        
//        GridMat descriptorsSubjTr (descriptorsTrFold, tagsTrFold, 1);
//        
//        modelSelection(descriptorsTrFold, tagsTrFold,
//                       params, goodnesses[k]);
//        
//        std::stringstream ss;
//        ss << "goodnesses_" << k << ".yml" << endl;
//        goodnesses[k].save(ss.str());
//    }
//    cout << endl;
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        GridMat tagsTrFold (tags, partitions, k, true);
        GridMat tagsTeFold (tags, partitions, k);
        GridMat descriptorsTrFold (descriptors, partitions, k, true);
        GridMat descriptorsTeFold (descriptors, partitions, k);
        
        GridMat descriptorsSubjTrFold (descriptorsTrFold, tagsTrFold, 1);
        
        GridPredictor<cv::EM> predictor;
        predictor.setData(descriptorsSubjTrFold);
       
        GridMat goodnesses;
        std::stringstream ss;
        ss << "goodnesses_" << k << ".yml" << endl;
        goodnesses.load(ss.str());
        
        vector<cv::Mat> bestParams;
        selectBestParameterCombination(gridExpandedParameters, m_hp, m_wp, params.size(), goodnesses, bestParams);

        predictor.setNumOfMixtures(bestParams[0]);
        predictor.setLoglikelihoodThreshold(bestParams[1]);
        predictor.train();
        
        GridMat predictionsTeFold, loglikelihoodsTeFold;
        predictor.predict(descriptorsTeFold, predictionsTeFold, loglikelihoodsTeFold);

        predictions.set(predictionsTeFold, partitions, k);
        loglikelihoods.set(loglikelihoodsTeFold, partitions, k);
    }
    cout << endl;
}

template<typename T>
void ModalityPrediction<cv::EM>::modelSelection(GridMat descriptors, GridMat tags,
                                                vector<vector<T> > params,
                                                GridMat& goodnesses)
{
    // Prepare parameters' combinations
    vector<vector<T> > gridExpandedParameters;
    expandParameters(params, m_hp * m_wp, gridExpandedParameters);
    
    // Partitionate the data in folds
    GridMat partitions;
    cvpartition(tags, m_modelSelecK, m_seed, partitions);
    
    // Instanciate a hp-by-wp GridMat of accuracies. A cell contains a matrix
    // being the rows the parameters' combinations and columns fold-runs
    GridMat accuracies;
    
    vector<GridPredictor<cv::EM> > predictors;
    
    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k << " ";
        
        // Get fold's data
        GridMat tagsTr (tags, partitions, k, true);
        GridMat tagsVal (tags, partitions, k);
        GridMat descriptorsTr (descriptors, partitions, k, true);
        GridMat descriptorsVal (descriptors, partitions, k);
        
        GridMat descriptorsSubjTr (descriptorsTr, tagsTr, 1); // subjects' training sample
        
        GridMat descriptorsSubjObjVal (descriptorsVal, tagsVal, -1, true);
        GridMat tagsSubjObjVal (tagsVal, tagsVal, -1, true);
        
        GridMat foldAccs; // results
        
        for (int m = 0; m < gridExpandedParameters.size(); m++)
        {
            // Create predictor and its parametrization
            GridPredictor<cv::EM> predictor;
            predictor.setData(descriptorsSubjTr); // unsupevised method, thus no tags
            
            cv::Mat nmixtures (m_hp, m_wp, cv::DataType<T>::type);
            cv::Mat loglikes (m_hp, m_wp, cv::DataType<T>::type);
            
            vector<cv::Mat> selectedParams;
            selectParameterCombination(gridExpandedParameters, m_hp, m_wp, params.size(), m, selectedParams);
            
            predictor.setNumOfMixtures(selectedParams[0]);
            predictor.setLoglikelihoodThreshold(selectedParams[1]);

            // Train
            predictor.train();
            
            // Test
            GridMat predictionsVal, loglikelihoodsVal;
            predictor.predict(descriptorsSubjObjVal, predictionsVal, loglikelihoodsVal);
            
            // Compute an accuracy measure
            cv::Mat accs; // (m_hp * m_wp) accuracies get by params combination in k-th fold
            accuracy(tagsSubjObjVal, predictionsVal, accs);
            
            GridMat paramsAccs (accs, m_hp, m_wp);
            foldAccs.vconcat(paramsAccs);
        }
        
        accuracies.hconcat(foldAccs);
    }
    cout << ") " << endl;
    
    //GridMat foldsMeanAcc;
    accuracies.mean(goodnesses, 1);

//    GridMat foldsArgmaxAcc;
//    foldsMeanAcc.argmax<float>(foldsArgmaxAcc);
//    selectParameters(expandedParams, foldsArgmaxAcc, selection);
}


//void ModalityPrediction<cv::EM>::expandParameters(vector<vector<double> > params, vector<vector<double> >& expandedParams)
//{
//    return ModalityPredictionBase<cv::EM>::expandParameters(params, expandedParams);
//}
//
//
//void ModalityPrediction<cv::EM>::expandParameters(vector<vector<double> > params, int ncells, vector<vector<double> >& gridExpandedParams)
//{
//    return ModalityPredictionBase<cv::EM>::expandParameters(params, ncells, gridExpandedParams);
//}
//
//void ModalityPrediction<cv::EM>::selectParameterCombination(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams)
//{
//    return ModalityPredictionBase<cv::EM>::selectParameterCombination(expandedParams, hp, wp, nparams, idx, selectedParams);
//}

void ModalityPrediction<cv::EM>::accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
{
    return ModalityPredictionBase<cv::EM>::accuracy(actuals, predictions, accuracies);
}


// Explicit template instanciation (to avoid linking errors)
template class ModalityPredictionBase<cv::EM>;
//template class ModalityPrediction<cv::EM>;