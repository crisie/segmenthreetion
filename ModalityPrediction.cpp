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
template void ModalityPrediction<cv::EM>::setData(ModalityGridData &data);
template void ModalityPrediction<cv::EM>::setModelSelection(int k, bool best);
template void ModalityPrediction<cv::EM>::setModelValidation(int k, int seed);

template void ModalityPrediction<cv::EM>::modelSelection<int>(GridMat descriptors, GridMat tags, vector<vector<int> > params, GridMat& goodnesses);
template void ModalityPrediction<cv::EM>::modelSelection<double>(GridMat descriptors, GridMat tags, vector<vector<double> > params, GridMat& goodnesses);
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


//
// ModalityPrediction<PredictorT>
//

template<typename PredictorT>
ModalityPrediction<PredictorT>::ModalityPrediction()
        : ModalityPredictionBase<PredictorT>()
{
    
}


//
// ModalityPrediction<cv::EM>
//

ModalityPrediction<cv::EM>::ModalityPrediction()
: ModalityPredictionBase<cv::EM>()
{
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

void ModalityPrediction<cv::EM>::compute(GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin)
{
    cv::Mat tags = m_data.getTagsMat();
    
    cv::Mat zeros = cv::Mat::zeros(tags.rows, tags.cols, tags.type());
    cv::Mat ones = cv::Mat::ones(tags.rows, tags.cols, tags.type());
    cv::Mat infinities (tags.rows, tags.cols, cv::DataType<float>::type);
    infinities.setTo(std::numeric_limits<float>::max());
    
    GridMat gtags;
    gtags.setTo(tags);
    GridMat gdescriptors = m_data.getDescriptors();
    GridMat gvalidnesses = m_data.getValidnesses();
    
    GridMat gpartitions;
    cvpartition(gtags, m_testK, m_seed, gpartitions);
    
    vector<GridPredictor<cv::EM> > predictors;
    
    vector<vector<int> > params, gridExpandedParameters;
    
    params.push_back(vector<int>(m_nmixtures.begin(), m_nmixtures.end()));
    params.push_back(vector<int>(m_logthresholds.begin(), m_logthresholds.end()));
    // create a list of parameters' variations
    expandParameters(params, m_hp * m_wp, gridExpandedParameters);
    
    
    cout << "Model selection CVs [" << m_testK << "]: " << endl;
    
    vector<GridMat> goodnesses(m_testK); // for instance: accuracies
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        // Index the k-th training
        GridMat validnessesTrFold (gvalidnesses, gpartitions, k, true);
        GridMat descriptorsTrFold (gdescriptors, gpartitions, k, true);
        GridMat tagsTrFold (gtags, gpartitions, k, true);
        
        // Within the k-th training partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTrFold = descriptorsTrFold.convertToDense(validnessesTrFold);
        GridMat validTagsTrFold = tagsTrFold.convertToDense(validnessesTrFold);
        
        modelSelection(validDescriptorsTrFold, validTagsTrFold,
                       params, goodnesses[k]);
        
        std::stringstream ss;
        ss << "gmm_goodnesses_" << k << ".yml" << endl;
        goodnesses[k].save(ss.str());
    }
    cout << endl;
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    GridMat individualPredictions;
    individualPredictions.setTo(zeros);
    
    loglikelihoods.setTo(zeros);
    distsToMargin.setTo(infinities);
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        // Index the k-th training and test partitions
        GridMat descriptorsTrFold (gdescriptors, gpartitions, k, true);
        GridMat descriptorsTeFold (gdescriptors, gpartitions, k);
        GridMat validnessesTrFold (gvalidnesses, gpartitions, k, true);
        GridMat validnessesTeFold (gvalidnesses, gpartitions, k);
        GridMat tagsTrFold (gtags, gpartitions, k, true);
        GridMat tagsTeFold (gtags, gpartitions, k);
        
        // Within the k-th training partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTrFold = descriptorsTrFold.convertToDense(validnessesTrFold);
        GridMat validTagsTrFold = tagsTrFold.convertToDense(validnessesTrFold);
        
        // Within the valid descriptors in the k-th training partition,
        // index the subject descriptors (tag == 1)
        GridMat validSubjectDescriptorsTrFold (validDescriptorsTrFold, validTagsTrFold, 1);
        
        
        GridPredictor<cv::EM> predictor(m_hp, m_wp);
       
        // Model selection information is kept on disk, reload it
        GridMat goodnesses;
        std::stringstream ss;
        ss << "gmm_goodnesses_" << k << ".yml" << endl;
        goodnesses.load(ss.str());
        
        // Train with the best parameter combination in average in a model
        // selection procedure within the training partition
        vector<cv::Mat> bestParams;
        selectBestParameterCombination(gridExpandedParameters, m_hp, m_wp, params.size(), goodnesses, bestParams);

        predictor.setNumOfMixtures(bestParams[0]);
        predictor.setLoglikelihoodThreshold(bestParams[1]);
        
        // Training phase
        
        predictor.train(validSubjectDescriptorsTrFold);
        
        // Predict phase
        
        // Within the k-th test partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTeFold = descriptorsTeFold.convertToDense(validnessesTeFold);
        
        GridMat validPredictionsTeFold, validLoglikelihoodsTeFold, validDistsToMargin;
        predictor.predict(validDescriptorsTeFold, validPredictionsTeFold, validLoglikelihoodsTeFold, validDistsToMargin);

        individualPredictions.set(validPredictionsTeFold.convertToSparse(validnessesTeFold), gpartitions, k);
        loglikelihoods.set(validLoglikelihoodsTeFold.convertToSparse(validnessesTeFold), gpartitions, k);
        distsToMargin.set(validDistsToMargin.convertToSparse(validnessesTeFold), gpartitions, k);
    }
    cout << endl;
    
    // Grid cells' consensus
    computeGridPredictionsConsensus(individualPredictions, distsToMargin, predictions); // predictions are consensued
}

template<typename T>
void ModalityPrediction<cv::EM>::modelSelection(GridMat descriptors, GridMat tags,
                                                vector<vector<T> > gridExpandedParams,
                                                GridMat& goodnesses)
{
    // Prepare parameters' combinations
//    vector<vector<T> > gridExpandedParameters;
//    expandParameters(params, m_hp * m_wp, gridExpandedParameters);
    
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

        GridMat descriptorsTr (descriptors, partitions, k, true);
        GridMat descriptorsVal (descriptors, partitions, k);
        GridMat tagsTr (tags, partitions, k, true);
        GridMat tagsVal (tags, partitions, k);
        
        GridMat descriptorsSubjTr (descriptorsTr, tagsTr, 1); // subjects' training sample
        
        GridMat descriptorsSubjObjVal (descriptorsVal, tagsVal, -1, true);
        GridMat tagsSubjObjVal (tagsVal, tagsVal, -1, true);
        
        GridMat foldAccs; // results
        
        for (int m = 0; m < gridExpandedParams.size(); m++)
        {
            // Create predictor and its parametrization
            GridPredictor<cv::EM> predictor (m_hp, m_wp);
            
            vector<cv::Mat> selectedParams;
            selectParameterCombination(gridExpandedParams, m_hp, m_wp,
                                       gridExpandedParams[0].size() / (m_hp * m_wp),
                                       m, selectedParams);
            
            predictor.setNumOfMixtures(selectedParams[0]);
            predictor.setLoglikelihoodThreshold(selectedParams[1]);

            // Train
            predictor.train(descriptorsSubjTr);
            
            // Test
            GridMat predictionsVal, loglikelihoodsVal, distsToMargin;
            predictor.predict(descriptorsSubjObjVal, predictionsVal, loglikelihoodsVal, distsToMargin);
            
            // Compute an accuracy measure
            cv::Mat accs; // (m_hp * m_wp) accuracies get by params combination in k-th fold
            accuracy(tagsSubjObjVal, predictionsVal, accs);
            
            GridMat paramsAccs (accs, m_hp, m_wp);
            foldAccs.vconcat(paramsAccs);
        }
        
        accuracies.hconcat(foldAccs);
    }
    cout << ") " << endl;
    
    accuracies.mean(goodnesses, 1);
}

void ModalityPrediction<cv::EM>::computeGridPredictionsConsensus(GridMat individualPredictions,
                                                                 GridMat distsToMargin,
                                                                 GridMat& consensusPredictions)
{
    cv::Mat tags = m_data.getTagsMat();
    
    cv::Mat consensus (tags.rows, tags.cols, tags.type());
    consensus.setTo(0);
    
    cv::Mat votes = individualPredictions.accumulate(); // accumulation expresses #"cells vote for subject"
    consensus.setTo(1, votes > ((m_hp * m_wp) / 2)); // if majority of cells vote for subject, it is
    
    // Deal with draws
    
    GridMat drawIndices;
    drawIndices.setTo(votes == ((m_hp * m_wp) / 2));
    
    GridMat drawnPredictions (individualPredictions, drawIndices); // index the subset of draws
    GridMat drawnDists (distsToMargin, drawIndices); // and their corresponding distances to margin
    
    GridMat negPredictions = (drawnPredictions == 0);
    GridMat posPredictions = (drawnPredictions == 1);
    
    GridMat negDists (drawnDists); // it is not a deep copy, so it is useful to set the same size
    GridMat posDists (drawnDists);
    negDists.setTo(0);
    posDists.setTo(0);
    
    drawnDists.copyTo(negDists, negPredictions);
    drawnDists.copyTo(posDists, posPredictions);
    
    cv::Mat avgNegDists, avgPosDists;
    cv::divide(negDists.accumulate(), negPredictions.accumulate(), avgNegDists);
    cv::divide(posDists.accumulate(), posPredictions.accumulate(), avgPosDists);
    
    GridMat consensusDrawnPredictions;
    consensusDrawnPredictions.setTo(avgPosDists > cv::abs(avgNegDists)); // not specifying the cell, copies to every cell
    
    consensusPredictions.set(consensusDrawnPredictions, drawIndices);
    
//    cv::Mat consensus;
//    consensus.setTo(0);
//    
//    cv::Mat votes; // the consensus for a grid takes into account the positive and negative cells' votes
//    individualPredictions.accumulate(votes); // 0 is object, 1 is subject. The accumulation expresses how many cells vote for subject
//    
//    cv::Mat positives = (votes > (m_hp * m_wp) / 2); // if the absolute majority of cells vote for subject, it is subject
//    ones.copyTo(consensus, positives);
//    
//    cv::Mat draws = (votes == (m_hp * m_wp) / 2); // draws must be undrawed by comparing the average dist to margin of the negative cells' votes and the positive cells' votes
//    cv::Mat negAvgDistToMargin, posAvgDistToMargin;
//    distsToMargin.standardize().biaverage(predictions, 0, negAvgDistToMargin, posAvgDistToMargin); // two averages, the one of the negative cells' predictions (==0) and the positive cells' predictions (>0);
//    ones.copyTo(consensusPredictions, posAvgDistToMargin < cv::abs(negAvgDistToMargin));
//    
//    predictions.setTo(consensusPredictions)
}