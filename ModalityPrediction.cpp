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
#include "CvExtraTools.h"

//
// ModalityPredictionBase
//

template<typename PredictorT>
ModalityPredictionBase<PredictorT>::ModalityPredictionBase()
: m_bModelSelection(true), m_bDimReduction(false), m_narrowSearchSteps(15)
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

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setDimensionalityReduction(float variance)
{
    m_bDimReduction = true;
    m_variance = variance;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::computeGridPredictionsConsensus(ModalityGridData data,
                                                                         GridMat predictions,
                                                                         GridMat distsToMargin,
                                                                         cv::Mat& consensusPredictions,
                                                                         cv::Mat& consensusDistsToMargin)
{
    cv::Mat tags = data.getTagsMat();
    
    consensusPredictions.create(tags.rows, 1, cv::DataType<int>::type);
    consensusDistsToMargin.create(tags.rows, 1, cv::DataType<float>::type);
    
    cv::Mat partitions;
    cvpartition(tags, m_testK, m_seed, partitions);
    
    for (int r = 0; r < tags.rows; r++)
    {
        int pos = 0;
        int neg = 0;
        float accPosDists = 0;
        float accNegDists = 0;
        for (int i = 0; i < data.getHp(); i++) for (int j = 0; j < data.getWp(); j++)
        {
            if (predictions.at<int>(i,j,r,0) == 0)
            {
                neg++;
                accNegDists += distsToMargin.at<float>(i,j,r,0);
            }
            else if (predictions.at<int>(i,j,r,0) == 1)
            {
                pos++;
                accPosDists += distsToMargin.at<float>(i,j,r,0);
            }
        }
        
        if (pos > neg)
        {
            consensusPredictions.at<int>(r,0) = 1;
            consensusDistsToMargin.at<float>(r,0) = accPosDists / pos;
        }
        else if (pos < neg)
        {
            consensusPredictions.at<int>(r,0) = 0;
            consensusDistsToMargin.at<float>(r,0) = accNegDists / neg;
        }
        else // pos == neg
        {
            if (accPosDists > abs(accNegDists)) // most confident towards positive classification
            {
                consensusPredictions.at<int>(r,0) = 1;
                consensusDistsToMargin.at<float>(r,0) = accPosDists / pos;
            }
            else
            {
                consensusPredictions.at<int>(r,0) = 0;
                consensusDistsToMargin.at<float>(r,0) = accNegDists / neg;
            }
        }
    }
    
//    cv::Mat tags = m_data.getTagsMat();
//    cv::Mat consensus (tags.rows, tags.cols, tags.type(), cv::Scalar(0));
//    int ncells = m_hp * m_wp;
//    
//    // Assume cells in the grid of individual predictions have all the same size
//    // so it is possible to accumulate the values in an element-wise fashion
//    cv::Mat votes = predictions.accumulate(); // accumulation expresses #"cells vote for subject"
//    
//    consensus.setTo(1, votes > (ncells / 2)); // if majority of cells vote for subject, it is
//    
//    // Deal with draws
//    // that is, the same number of cells voting positively and negatively
//    
//    GridMat drawIndices;
//    drawIndices.setTo(votes == (ncells / 2));
//    
//    GridMat drawnPredictions (predictions, drawIndices); // index the subset of draws
//    GridMat drawnDists (distsToMargin, drawIndices); // and their corresponding distances to margin
//    
//    GridMat negPredictions = (drawnPredictions == 0) / 255;
//    GridMat posPredictions = (drawnPredictions == 1) / 255;
//    
//    GridMat negDists, posDists;
//    drawnDists.copyTo(negDists, negPredictions);
//    drawnDists.copyTo(posDists, posPredictions);
//    
//    cv::Mat_<float> avgNegDists, avgPosDists;
//    avgNegDists = negDists.accumulate() / (ncells / 2);
//    avgPosDists = posDists.accumulate() / (ncells / 2);
//    
//    cv::Mat aux = (avgPosDists > cv::abs(avgNegDists));
//    cvx::setMat(aux / 255, consensus, votes == (ncells / 2));
//    
//    // When consensuated, all the cells in the grid of consensus are exactly the same
//    consensusPredictions.setTo(consensus);
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

void ModalityPrediction<cv::EM>::compute(GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin, GridMat& accuracies)
{
    cv::Mat tags = m_data.getTagsMat();
    
//    cv::Mat zeros = cv::Mat::zeros(tags.rows, tags.cols, tags.type());
//    cv::Mat ones = cv::Mat::ones(tags.rows, tags.cols, tags.type());
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
		cout << m_data.getModality() << " model selection CVs [" << m_testK << "]: " << endl;
        
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
            
            GridMat goodnesses;
            modelSelection(validDescriptorsTrFold, validTagsTrFold,
                           gridExpandedParameters, goodnesses);


            std::stringstream ss;
            ss << m_data.getModality() << "_models_goodnesses_" << k << ".yml";
            goodnesses.save(ss.str());
        }
        cout << endl;
    }
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
     
    GridMat individualPredictions;
    
    individualPredictions.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<int>::type));
    loglikelihoods.setTo(negInfinities);
    distsToMargin.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<float>::type));
    
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
        ss << m_data.getModality() << "_models_goodnesses_" << k << ".yml";
        goodnesses.load(ss.str());
        
        // Train with the best parameter combination in average in a model
        // selection procedure within the training partition
        vector<cv::Mat> bestParams;
        selectBestParameterCombination(gridExpandedParameters, m_hp, m_wp, params.size(), goodnesses, bestParams);

        predictor.setNumOfMixtures(bestParams[0]);
        predictor.setLoglikelihoodThreshold(bestParams[1]);
        
        if (m_bDimReduction)
            predictor.setDimensionalityReduction(cv::Mat(m_hp, m_wp, cv::DataType<double>::type, cv::Scalar(m_variance)));

        // Training phase
        
        predictor.train(validSubjectDescriptorsTrFold);
        
        // Predict phase
        
        // Within the k-th test partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTeFold = descriptorsTeFold.convertToDense(validnessesTeFold);
        
        GridMat validPredictionsTeFold, validLoglikelihoodsTeFold, validDistsToMargin;
        predictor.predict(validDescriptorsTeFold, validPredictionsTeFold, validLoglikelihoodsTeFold, validDistsToMargin);
        
        // Compute a goodness measure of this individual prediction (accuracy)
        GridMat validTagsTeFold = tagsTeFold.convertToDense(validnessesTeFold);
        GridMat validTagsSubjObjTeFold (validTagsTeFold, validTagsTeFold, -1, true);
        GridMat validPredictionsSubjObjTeFold (validPredictionsTeFold, validTagsTeFold, -1, true);
        
        cv::Mat accsFold;
        accuracy(validTagsSubjObjTeFold, validPredictionsSubjObjTeFold, accsFold);
        GridMat aux (accsFold, m_hp, m_wp);
        accuracies.vconcat(aux);

        // Store other results apart from the goodness
        individualPredictions.set(validPredictionsTeFold.convertToSparse(validnessesTeFold), gpartitions, k);
        loglikelihoods.set(validLoglikelihoodsTeFold.convertToSparse(validnessesTeFold), gpartitions, k);
        distsToMargin.set(validDistsToMargin.convertToSparse(validnessesTeFold), gpartitions, k);
    }
    cout << endl;

    // Grid cells' consensus
    // TODO: move this function to the fusion part
    // computeGridPredictionsConsensus(individualPredictions, distsToMargin, predictions); // predictions are consensued
    predictions = individualPredictions;
}

template<typename T>
//void ModalityPrediction<cv::EM>::modelSelection(cv::Mat descriptors, cv::Mat tags,
//                                                vector<vector<T> > gridExpandedParams,
//                                                cv::Mat& goodness)
void ModalityPrediction<cv::EM>::modelSelection(GridMat descriptors, GridMat tags,
                                                vector<vector<T> > gridExpandedParams,
                                                GridMat& goodnesses)
{
    GridMat partitions;
    cvpartition(tags, m_modelSelecK, m_seed, partitions);
    
    GridMat accuracies;
    accuracies.setTo(cv::Mat(gridExpandedParams.size(), m_modelSelecK, cv::DataType<float>::type));
    
    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        // Get fold's data
        cout << k;
        
        GridMat descriptorsTr (descriptors, partitions, k, true);
        GridMat descriptorsVal (descriptors, partitions, k);
        GridMat tagsTr (tags, partitions, k, true);
        GridMat tagsVal (tags, partitions, k);
        
        GridMat descriptorsSubjTr (descriptorsTr, tagsTr, 1); // subjects' training sample
        
        GridMat descriptorsSubjObjVal (descriptorsVal, tagsVal, -1, true);
        GridMat tagsSubjObjVal (tagsVal, tagsVal, -1, true);
        
        GridMat accsFold; // results
        
        cv::EM predictor;
        for (int i = 0; i < m_data.getHp(); i++) for (int j = 0; j < m_data.getWp(); j++)
        {
            cv::PCA pca; // if m_bDimReduction is false, this variable is not used anymore
            
            cv::Mat cellDescriptorsSubjTr;
            if (!m_bDimReduction)
                cellDescriptorsSubjTr = descriptorsSubjTr.at(i,j);
            else
                cvx::computePCA(descriptorsSubjTr.at(i,j), pca,
                                cellDescriptorsSubjTr, CV_PCA_DATA_AS_ROW, m_variance);
            
            for (int m = 0; m < gridExpandedParams.size(); m++)
            {
                // Create predictor and its parametrization
                int nclusters = predictor.get<int>("nclusters");
                if (gridExpandedParams[m][0] != nclusters)
                {
                    predictor.set("nclusters", gridExpandedParams[m][0]);
                
                    // Train
                    predictor.train(cellDescriptorsSubjTr);
                }
                
                // Test
                cv::Mat_<float> loglikelihoods;
                for (int d = 0; d < descriptorsSubjObjVal.at(i,j).rows; d++)
                {
                    cv::Mat descriptor = descriptorsSubjObjVal.at(i,j).row(d);
                    
                    if (m_bDimReduction)
                        descriptor = pca.project(descriptor);
                    
                    cv::Vec2d res = predictor.predict(descriptor);
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
                float acc = accuracy(tagsSubjObjVal.at(i,j), predictions);
                
                accsFold.at(i,j).push_back(acc); // element to row
            }
            
            accsFold.at(i,j).copyTo(accuracies.at(i,j).col(k));
        }
    }
    cout << ")" << endl;
    
    accuracies.mean(goodnesses, 1);
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
    
    objDistribution.create(nbins, 1, cv::DataType<int>::type);
    sbjDistribution.create(nbins, 1, cv::DataType<int>::type);
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
//            cout << m << " ";
            
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


//
// ModalityPrediction<cv::Mat>
//

ModalityPrediction<cv::Mat>::ModalityPrediction()
: ModalityPredictionBase<cv::Mat>()
{
}

void ModalityPrediction<cv::Mat>::setPositiveClassificationRatios(float m)
{
    m_ratios.clear();
    m_ratios.push_back(m);
}

void ModalityPrediction<cv::Mat>::setPositiveClassificationRatios(vector<float> m)
{
    m_ratios = m;
}

void ModalityPrediction<cv::Mat>::setScoreThresholds(float t)
{
    m_scores.clear();
    m_scores.push_back(t);
}

void ModalityPrediction<cv::Mat>::setScoreThresholds(vector<float> t)
{
    m_scores = t;
}

void ModalityPrediction<cv::Mat>::compute(GridMat& individualPredictions, GridMat& gscores, GridMat& distsToMargin, GridMat& accuracies)
{
    cv::Mat tags = m_data.getTagsMat();
    
    cv::Mat negInfinities (tags.rows, tags.cols, cv::DataType<float>::type);
    cv::Mat posInfinities (tags.rows, tags.cols, cv::DataType<float>::type);
    negInfinities.setTo(-std::numeric_limits<float>::infinity());
    posInfinities.setTo(std::numeric_limits<float>::infinity());
    
    GridMat gtags;
    gtags.setTo(tags);
    GridMat gvalidnesses = m_data.getValidnesses();
    
    cv::Mat partitions;
    cvpartition(tags, m_testK, m_seed, partitions);
    GridMat gpartitions (partitions);
    
    // create a list of parameters' variations
    vector<vector<float> > params;
    vector<float> ratios (m_ratios.begin(), m_ratios.end());
    vector<float> scoreThresholds (m_scores.begin(), m_scores.end());
    params.push_back(ratios);
    params.push_back(scoreThresholds);
    
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
    if (m_bModelSelection)
    {
		cout << m_data.getModality() << " model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " ";
            
            GridMat coarseGoodnesses (m_hp, m_wp);
            GridMat narrowGoodnesses (m_hp, m_wp);
            
            for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
            {
                cv::Mat validnessesTr = cvx::indexMat(gvalidnesses.at(i,j), partitions != k);
                cv::Mat tagsTr = cvx::indexMat(tags, partitions != k);
                cv::Mat indicesTr = cvx::indexMat(cvx::linspace(0, tags.rows), partitions != k);
                
                cv::Mat validIndicesTr = cvx::indexMat(indicesTr, validnessesTr);
                cv::Mat validTagsTr = cvx::indexMat(tagsTr, validnessesTr);

                // Coarse search
                cv::Mat coarseCellGoodnesses;
                modelSelection<float>(validIndicesTr, validTagsTr, i, j,
                                      coarseExpandedParameters, coarseCellGoodnesses);
                
                // Narrow search
                cv::Mat narrowExpandedParameters;
                int discretes[] = {0,0};
                narrow<float>(coarseExpandedParameters, coarseCellGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
                
                cv::Mat narrowCellGoodnesses;
                modelSelection<float>(validIndicesTr, validTagsTr, i, j,
                                      narrowExpandedParameters, narrowCellGoodnesses);
                
                cv::hconcat(coarseExpandedParameters, coarseCellGoodnesses, coarseCellGoodnesses);
                cv::hconcat(narrowExpandedParameters, narrowCellGoodnesses, narrowCellGoodnesses);
                
                coarseGoodnesses.assign(coarseCellGoodnesses, i, j);
                narrowGoodnesses.assign(narrowCellGoodnesses, i, j);
            }
            
            std::stringstream coarsess;
            coarsess << m_data.getModality() << "_models_coarse-goodnesses_" << k << ".yml";
            coarseGoodnesses.save(coarsess.str());
            
            std::stringstream narrowss;
            narrowss << m_data.getModality() << "_models_narrow-goodnesses_" << k << ".yml";
            narrowGoodnesses.save(narrowss.str());
        }
        cout << endl;
    }
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    individualPredictions.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<int>::type));
    gscores.setTo(negInfinities);
    distsToMargin.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<float>::type));
    
    accuracies.create(m_hp, m_wp);
    accuracies.setTo(cv::Mat(m_testK,1,cv::DataType<float>::type));
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        // Index the k-th test partitions
        cv::Mat indicesTeFold = cvx::indexMat(cvx::linspace(0, tags.rows), partitions == k);
//        GridMat validnessesTeFold (gvalidnesses, gpartitions, k);
//        GridMat tagsTeFold (gtags, gpartitions, k);
        cv::Mat tagsTeFold = cvx::indexMat(tags, partitions == k);
        
        // Model selection information is kept on disk, reload it
        GridMat goodnesses;
        std::stringstream ss;
        ss << m_data.getModality() << "_models_narrow-goodnesses_" << k << ".yml";
        goodnesses.load(ss.str());
        
        // Train with the best parameter combination in average in a model
        // selection procedure within the training partition
        vector<cv::Mat> bestParams;
        selectBestParameterCombination<float>(goodnesses, bestParams);
        
//        // DEBUG
//        for (int i = 0; i < goodnesses.at(0,0).rows; i++)
//        {
//            cout << goodnesses.at(0,0).row(i) << endl;
//        }
//        cout << bestParams[0].at<float>(0,0) << " " << bestParams[1].at<float>(0,0) << endl;
        
        
        // Predict phase
        
        
        
        for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
        {
            cv::Mat validnessesTe = cvx::indexMat(gvalidnesses.at(i,j), partitions == k);
            
            cv::Mat validIndicesTe = cvx::indexMat(indicesTeFold, validnessesTe);
            cv::Mat validTagsTe = cvx::indexMat(tagsTeFold, validnessesTe);
        
            cv::Mat validPredictions (validIndicesTe.rows, 1, cv::DataType<int>::type);
            cv::Mat validScores      (validIndicesTe.rows, 1, cv::DataType<float>::type);
            cv::Mat validDistances   (validIndicesTe.rows, 1, cv::DataType<float>::type);
            
            float ratio = bestParams[0].at<float>(i,j);
            float score = bestParams[1].at<float>(i,j);
            
            for (int d = 0; d < validIndicesTe.rows; d++)
            {
                int idx = validIndicesTe.at<int>(d,0);
                
                cv::Mat_<float> cell = m_data.getGridFrame(idx).at(i,j);
                cv::Mat cellMask = m_data.getGridMask(idx).at(i,j);
                
                cv::Mat_<float> aux;
                cv::threshold(cell, aux, score, 1, cv::THRESH_BINARY);
                CvScalar mean = cv::mean(aux, cellMask);
                
                validPredictions.at<int>(d,0) = (mean.val[0] > ratio);
                validScores.at<float>(d,0) = (mean.val[0]);
                validDistances.at<float>(d,0) = score - mean.val[0];
            }
            
            cv::Scalar mean, stddev;
            cv::meanStdDev(validDistances, mean, stddev);
            cv::Mat_<float> stdValidDistances = (validDistances - mean.val[0]) / stddev.val[0];
            
            cv::Mat predictions, distances;
            cv::Mat scores (indicesTeFold.rows, 1, cv::DataType<float>::type);
            scores.setTo(cv::mean(validScores).val[0]); // valids' mean
            
            cout << validTagsTe << endl;
            cout << validPredictions << endl;
            accuracies.at<float>(i,j,k,0) = accuracy(validTagsTe, validPredictions);
            
            cvx::setMat(validPredictions, predictions, validnessesTe); // not indexed by validnessesTe take 0
            cvx::setMat(validScores, scores, validnessesTe); // not indexed by validnessesTe take valids' mean
            cvx::setMat(stdValidDistances, distances, validnessesTe); // not indexed by validnessesTe take 0
            
            cvx::setMat(predictions, individualPredictions.at(i,j), partitions == k);
            cvx::setMat(scores, gscores.at(i,j), partitions == k);
            cvx::setMat(distances, distsToMargin.at(i,j), partitions == k);
        }
    }
    cout << endl;
    
    // Grid cells' consensus
    //computeGridPredictionsConsensus(individualPredictions, distsToMargin, predictions); // predictions are consensued
}

template<typename T>
void ModalityPrediction<cv::Mat>::modelSelection(cv::Mat indices, cv::Mat tags,
                                                 unsigned int i, unsigned int j,
                                                 cv::Mat expandedParams,
                                                 cv::Mat& goodness)
{
    cv::Mat indsSubjObj  = cvx::indexMat(indices, tags >= 0);
    cv::Mat tagsSubjObj = cvx::indexMat(tags, tags >= 0);

    goodness.release();
    goodness.create(expandedParams.rows, 1, cv::DataType<float>::type); // results

    for (int m = 0; m < expandedParams.rows; m++)
    {
        float ratio = expandedParams.at<T>(m,0);
        float score = expandedParams.at<T>(m,1);
        
        cv::Mat predictions (tagsSubjObj.rows, 1, cv::DataType<int>::type);
        predictions.setTo(0);
        for (int k = 0; k < indsSubjObj.rows; k++)
        {
            int idx = indsSubjObj.at<int>(k,0);
            
            cv::Mat_<float> cell = m_data.getGridFrame(idx).at(i,j);
            cv::Mat cellMask = m_data.getGridMask(idx).at(i,j);
            
            cv::Mat_<float> thrCell;
            cv::threshold(cell, thrCell, score, 1, cv::THRESH_BINARY);
            CvScalar mean = cv::mean(thrCell, cellMask);
            
            predictions.at<int>(k,0) = (mean.val[0] > ratio);
        }
        
        // Compute an accuracy measure
        goodness.at<float>(m,0) = accuracy(tagsSubjObj, predictions);
    }
}


// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void ModalityPredictionBase<cv::EM>::setData(ModalityGridData &data);
template void ModalityPredictionBase<cv::EM>::setModelSelection(bool flag);
template void ModalityPredictionBase<cv::EM>::setModelSelectionParameters(int k, bool best);
template void ModalityPredictionBase<cv::EM>::setValidationParameters(int k, int seed);
template void ModalityPredictionBase<cv::EM>::setDimensionalityReduction(float variance);
template void ModalityPredictionBase<cv::EM>::computeGridPredictionsConsensus(ModalityGridData data,
                                                                              GridMat predictions,
                                                                              GridMat distsToMargin,
                                                                              cv::Mat& consensusPredictions,
                                                                              cv::Mat& consensusDistsToMargin);

template void ModalityPredictionBase<cv::Mat>::setData(ModalityGridData &data);
template void ModalityPredictionBase<cv::Mat>::setModelSelection(bool flag);
template void ModalityPredictionBase<cv::Mat>::setModelSelectionParameters(int k, bool best);
template void ModalityPredictionBase<cv::Mat>::setValidationParameters(int k, int seed);
template void ModalityPredictionBase<cv::Mat>::computeGridPredictionsConsensus(ModalityGridData data,
                                                                               GridMat predictions,
                                                                               GridMat distsToMargin,
                                                                               cv::Mat& consensusPredictions,
                                                                               cv::Mat& consensusDistsToMargin);

template void ModalityPrediction<cv::EM>::modelSelection<int>(GridMat descriptors, GridMat tags, vector<vector<int> > params, GridMat& goodness);
template void ModalityPrediction<cv::EM>::modelSelection<float>(GridMat descriptors, GridMat tags, vector<vector<float> > params, GridMat& goodness);
template void ModalityPrediction<cv::EM>::modelSelection<double>(GridMat descriptors, GridMat tags, vector<vector<double> > params, GridMat& goodness);

template void ModalityPrediction<cv::Mat>::modelSelection<int>(cv::Mat indices, cv::Mat tags,
                                                               unsigned int i, unsigned int j,
                                                               cv::Mat expandedParams,
                                                               cv::Mat& goodness);
template void ModalityPrediction<cv::Mat>::modelSelection<float>(cv::Mat indices, cv::Mat tags,
                                                                 unsigned int i, unsigned int j,
                                                                 cv::Mat expandedParams,
                                                                 cv::Mat& goodness);
template void ModalityPrediction<cv::Mat>::modelSelection<double>(cv::Mat indices, cv::Mat tags,
                                                                  unsigned int i, unsigned int j,
                                                                  cv::Mat expandedParams,
                                                                  cv::Mat& goodness);
// -----------------------------------------------------------------------------
