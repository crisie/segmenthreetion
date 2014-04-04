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

template<typename PredictorT>
ModalityPredictionBase<PredictorT>::ModalityPredictionBase() : m_bModelSelection(true)
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
void ModalityPredictionBase<PredictorT>::setModelSelection(bool flag)
{
    m_bModelSelection = flag;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setModelSelectionParameters(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setValidationParameters(int k, int seed)
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

void ModalityPrediction<cv::EM>::setLoglikelihoodThresholds(float t)
{
    m_logthresholds.clear();
    m_logthresholds.push_back(t);
}

void ModalityPrediction<cv::EM>::setLoglikelihoodThresholds(vector<float> t)
{
    m_logthresholds = t;
}

void ModalityPrediction<cv::EM>::compute(GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin)
{
    cv::Mat tags = m_data.getTagsMat();
    
    cv::Mat zeros = cv::Mat::zeros(tags.rows, tags.cols, tags.type());
    cv::Mat ones = cv::Mat::ones(tags.rows, tags.cols, tags.type());
    cv::Mat negInfinities (tags.rows, tags.cols, cv::DataType<float>::type);
    cv::Mat posInfinities (tags.rows, tags.cols, cv::DataType<float>::type);
    
    negInfinities.setTo(-std::numeric_limits<float>::infinity());
    posInfinities.setTo(std::numeric_limits<float>::infinity());
    
    GridMat gtags;
    gtags.setTo(tags);
    GridMat gdescriptors = m_data.getDescriptors();
    GridMat gvalidnesses = m_data.getValidnesses();
    
    GridMat gpartitions;
    cvpartition(gtags, m_testK, m_seed, gpartitions);
    
    // create a list of parameters' variations
    vector<vector<float> > params, gridExpandedParameters;
    vector<float> nmixtures (m_nmixtures.begin(), m_nmixtures.end());
    vector<float> nlogthresholds (m_logthresholds.begin(), m_logthresholds.end());
    params.push_back(nmixtures);
    params.push_back(nlogthresholds);
    expandParameters(params, gridExpandedParameters);
    
    if (m_bModelSelection)
    {
        cout << "Model selection CVs [" << m_testK << "]: " << endl;
        
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
            
            GridMat goodnesses (m_hp, m_wp);
            for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
            {
                cv::Mat goodness;
                modelSelection(validDescriptorsTrFold.at(i,j), validTagsTrFold.at(i,j),
                               gridExpandedParameters, goodness);
                goodnesses.assign(goodness, i, j);
            }

            std::stringstream ss;
            ss << m_data.getModality() << "_models_goodnesses_" << k << ".yml" << endl;
            goodnesses.save(ss.str());
        }
        cout << endl;
    }
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    GridMat individualPredictions;
    
    individualPredictions.setTo(zeros);
    loglikelihoods.setTo(negInfinities);
    distsToMargin.setTo(posInfinities);
    
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
        ss << m_data.getModality() << "_models_goodnesses_" << k << ".yml" << endl;
        goodnesses.load(ss.str());
        
        // Train with the best parameter combination in average in a model
        // selection procedure within the training partition
        vector<cv::Mat> bestParams;
        selectBestParameterCombination(gridExpandedParameters, m_hp, m_wp, params.size(), goodnesses, bestParams);

        predictor.setNumOfMixtures(bestParams[0]);
        predictor.setLoglikelihoodThreshold(bestParams[1]);
        cout << bestParams[0] << endl;
        cout << bestParams[1] << endl;
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
void ModalityPrediction<cv::EM>::modelSelection(cv::Mat descriptors, cv::Mat tags,
                                                vector<vector<T> > gridExpandedParams,
                                                cv::Mat& goodness)
{
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(tags, m_modelSelecK, m_seed, partitions);
    cout << tags << endl;
    cout << partitions << endl;
    
    cv::Mat accuracies (gridExpandedParams.size(), m_modelSelecK, cv::DataType<float>::type);
    
    for (int k = 0; k < m_modelSelecK; k++)
    {
        // Get fold's data

        cv::Mat descriptorsTr = cvx::indexMat(descriptors, partitions != k);
        cv::Mat descriptorsVal = cvx::indexMat(descriptors, partitions == k);
        cv::Mat tagsTr = cvx::indexMat(tags, partitions != k);
        cv::Mat tagsVal = cvx::indexMat(tags, partitions == k);
        
        cv::Mat descriptorsSubjTr = cvx::indexMat(descriptorsTr, tagsTr == 1); // subjects' training sample
        
        cv::Mat descriptorsSubjObjVal = cvx::indexMat(descriptorsVal, tagsVal >= 0);
        cv::Mat tagsSubjObjVal = cvx::indexMat(tagsVal, tagsVal >= 0);
        
        cv::Mat foldAccs; // results
        
        cv::EM predictor;
        for (int m = 0; m < gridExpandedParams.size(); m++)
        {
            // Create predictor and its parametrization
            int nclusters = predictor.get<int>("nclusters");
            if (gridExpandedParams[m][0] != nclusters)
            {
                predictor.set("nclusters", gridExpandedParams[m][0]);
            
                // Train
                predictor.train(descriptorsSubjTr);
            }
            
            // Test
            cv::Mat_<float> loglikelihoods;
            for (int i = 0; i < descriptorsSubjObjVal.rows; i++)
            {
                cv::Vec2d res = predictor.predict(descriptorsSubjObjVal.row(i));
                loglikelihoods.push_back(res.val[0]);
            }

            // Standardized loglikelihoods
            cv::Mat_<float> stdLoglikelihoods;
            cv::Scalar mean, stddev;
            cv::meanStdDev(loglikelihoods, mean, stddev);
            stdLoglikelihoods = (loglikelihoods - mean.val[0]) / stddev.val[0];
            
            // Predictions evaluation comparing the standardized loglikelihoods to a threshold,
            // loglikelihoods over threshold are considered subject (1)
            cv::Mat predictions;
            cv::threshold(stdLoglikelihoods, predictions, gridExpandedParams[m][1], 1, CV_THRESH_BINARY);
            predictions.convertTo(predictions, cv::DataType<int>::type);
            
            // Compute an accuracy measure
            float acc = accuracy(tagsSubjObjVal, predictions);
            
            foldAccs.push_back(acc); // element to row
        }
        
        foldAccs.copyTo(accuracies.col(k));
    }
    
    cv::reduce(accuracies, goodness, 1, CV_REDUCE_AVG); // column of row-wise averages
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
    cv::Mat accNegPredictions, accPosPredictions;
    negPredictions.accumulate().convertTo(accNegPredictions, CV_32F);
    posPredictions.accumulate().convertTo(accPosPredictions, CV_32F);
    cv::divide(negDists.accumulate(), accNegPredictions, avgNegDists);
    cv::divide(posDists.accumulate(), accPosPredictions, avgPosDists);
    
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

void ModalityPrediction<cv::EM>::computeLoglikelihoodsDistribution(int nbins, double min, double max, cv::Mat& sbjDistribution, cv::Mat& objDistribution)
{
    cv::Mat tags = m_data.getTagsMat();
    GridMat gtags;
    gtags.setTo(tags);
    
    GridMat gdescriptors = m_data.getDescriptors();
    GridMat gvalidnesses = m_data.getValidnesses();
    
    GridMat gpartitions;
    cvpartition(gtags, m_testK, m_seed, gpartitions);
    
    vector<vector<int> > params, gridExpandedParams;
    
    params.push_back(vector<int>(m_nmixtures.begin(), m_nmixtures.end()));
    // create a list of parameters' variations
    expandParameters(params, gridExpandedParams);
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    objDistribution.create(nbins, 1, cv::DataType<float>::type);
    sbjDistribution.create(nbins, 1, cv::DataType<float>::type);
    objDistribution.setTo(0);
    sbjDistribution.setTo(0);
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " (";
        
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
        
        GridMat validDescriptorsTeFold = descriptorsTeFold.convertToDense(validnessesTeFold);
        GridMat validTagsTeFold = tagsTeFold.convertToDense(validnessesTeFold);
        
        // Within the valid descriptors in the k-th training partition,
        // index the subject descriptors (tag == 1)
        GridMat validSbjDescriptorsTrFold (validDescriptorsTrFold, validTagsTrFold, 1);
        
        GridMat validObjDescriptorsTeFold (validDescriptorsTeFold, validTagsTeFold, 0);
        GridMat validSbjDescriptorsTeFold (validDescriptorsTeFold, validTagsTeFold, 1);
        
        
        GridPredictor<cv::EM> predictor(m_hp, m_wp);
        
        for (int m = 0; m < gridExpandedParams.size(); m++)
        {
            cout << m << " ";
            
            cv::Mat numOfMixtures;
            numOfMixtures.setTo(gridExpandedParams[m][0]);
            predictor.setNumOfMixtures(numOfMixtures);
            
            // Training phase
            
            predictor.train(validSbjDescriptorsTrFold);
            
            // Predict phase
            
            GridMat validObjLoglikelihoodsTeFold, validSbjLoglikelihoodsTeFold;
            predictor.predict(validObjDescriptorsTeFold, validObjLoglikelihoodsTeFold);
            predictor.predict(validSbjDescriptorsTeFold, validSbjLoglikelihoodsTeFold);
            
            GridMat valObjLoglikelihoodsTeFoldHist = validObjLoglikelihoodsTeFold.historize(nbins, min, max);
            GridMat valSbjLoglikelihoodsTeFoldHist = validSbjLoglikelihoodsTeFold.historize(nbins, min, max);
            
            cv::Mat objHist = valObjLoglikelihoodsTeFoldHist.accumulate();
            cv::Mat sbjHist = valSbjLoglikelihoodsTeFoldHist.accumulate();
            
            cv::add(objDistribution, objHist, objDistribution);
            cv::add(sbjDistribution, sbjHist, sbjDistribution);
        }
        
        cout << ")" << endl;
    }
    cout << endl;
}



// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void ModalityPrediction<cv::EM>::setData(ModalityGridData &data);
template void ModalityPrediction<cv::EM>::setModelSelection(bool flag);
template void ModalityPrediction<cv::EM>::setModelSelectionParameters(int k, bool best);
template void ModalityPrediction<cv::EM>::setValidationParameters(int k, int seed);

template void ModalityPrediction<cv::EM>::modelSelection<int>(cv::Mat descriptors, cv::Mat tags, vector<vector<int> > params, cv::Mat& goodness);
template void ModalityPrediction<cv::EM>::modelSelection<float>(cv::Mat descriptors, cv::Mat tags, vector<vector<float> > params, cv::Mat& goodness);
template void ModalityPrediction<cv::EM>::modelSelection<double>(cv::Mat descriptors, cv::Mat tags, vector<vector<double> > params, cv::Mat& goodness);
// -----------------------------------------------------------------------------
