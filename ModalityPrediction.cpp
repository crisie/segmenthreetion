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
#include "em.h" // hack

#include <boost/timer.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

//
// ModalityPredictionBase
//

template<typename PredictorT>
ModalityPredictionBase<PredictorT>::ModalityPredictionBase()
: m_bModelSelection(true), m_bDimReduction(false), m_narrowSearchSteps(15),
  m_bTrainMirrored(false)
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
void ModalityPredictionBase<PredictorT>::setModelSelectionParameters(int k, bool bGlobalBest)
{
    m_modelSelecK = k;
    m_bGlobalBest = bGlobalBest;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setValidationParameters(int k)
{
    m_testK = k;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setDimensionalityReduction(float variance)
{
    m_bDimReduction = true;
    m_variance = variance;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setTrainMirrored(bool flag)
{
    m_bTrainMirrored = flag;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setPredictions(GridMat predictionsGrid)
{
    m_PredictionsGrid = predictionsGrid;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::setDistsToMargin(GridMat distsToMarginGrid)
{
    m_DistsToMarginGrid = distsToMarginGrid;
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::computeGridConsensusPredictions(cv::Mat& consensusPredictions,
                                                                         cv::Mat& consensusDistsToMargin)
{
    cv::Mat tags = m_data.getTagsMat();
    
    consensusPredictions.create(tags.rows, 1, cv::DataType<int>::type);
    consensusDistsToMargin.create(tags.rows, 1, cv::DataType<float>::type);
    
//    cv::Mat partitions;
//    cvpartition(tags, m_testK, m_seed, partitions);
    
    for (int r = 0; r < tags.rows; r++)
    {
        int pos = 0;
        int neg = 0;
        float accPosDists = 0;
        float accNegDists = 0;
        for (int i = 0; i < m_data.getHp(); i++) for (int j = 0; j < m_data.getWp(); j++)
        {
            if (m_PredictionsGrid.at<int>(i,j,r,0) == 0)
            {
                neg++;
                accNegDists += m_DistsToMarginGrid.at<float>(i,j,r,0);
            }
            else if (m_PredictionsGrid.at<int>(i,j,r,0) == 1)
            {
                pos++;
                accPosDists += m_DistsToMarginGrid.at<float>(i,j,r,0);
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

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::getAccuracy(cv::Mat predictions, cv::Mat& accuracies)
{
    accuracy(m_data.getTagsMat(), predictions, m_data.getPartitions(), accuracies);
}

template<typename PredictorT>
void ModalityPredictionBase<PredictorT>::getAccuracy(GridMat predictions, GridMat& accuracies)
{
    accuracy(m_data.getTagsMat(), predictions, m_data.getPartitions(), accuracies);
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
// ModalityPrediction<cv::EM40>
//

ModalityPrediction<cv::EM40>::ModalityPrediction()
: ModalityPredictionBase<cv::EM40>()
{
}

void ModalityPrediction<cv::EM40>::setNumOfMixtures(int m)
{
    m_nmixtures.clear();
    m_nmixtures.push_back(m);
}

void ModalityPrediction<cv::EM40>::setNumOfMixtures(vector<int> m)
{
    m_nmixtures = m;
}

void ModalityPrediction<cv::EM40>::setEpsilons(float eps)
{
    m_epsilons.clear();
    m_epsilons.push_back(eps);
}

void ModalityPrediction<cv::EM40>::setEpsilons(vector<float> eps)
{
    m_epsilons = eps;
}

void ModalityPrediction<cv::EM40>::setLoglikelihoodThresholds(float t)
{
    m_logthresholds.clear();
    m_logthresholds.push_back(t);
}

void ModalityPrediction<cv::EM40>::setLoglikelihoodThresholds(vector<float> t)
{
    m_logthresholds = t;
}

void ModalityPrediction<cv::EM40>::predict(GridMat& predictionsGrid, GridMat& loglikelihoodsGrid, GridMat& distsToMarginGrid)
{
    cv::Mat tags = m_data.getTagsMat();
    
    GridMat tagsGrid;
    tagsGrid.setTo(tags);
    GridMat descriptorsGrid = m_data.getDescriptors();
    GridMat validnessesGrid = m_data.getValidnesses();
    
    GridMat partitionsGrid;
    cv::Mat partition = m_data.getPartitions();
    partitionsGrid.setTo(partition);
    
    // create a list of parameters' variations
    vector<vector<float> > params, gridExpandedParameters;
    vector<float> nmixtures (m_nmixtures.begin(), m_nmixtures.end());
    vector<float> epsilons (m_epsilons.begin(), m_epsilons.end());
    vector<float> nlogthresholds (m_logthresholds.begin(), m_logthresholds.end());
    params.push_back(nmixtures);
    params.push_back(epsilons);
    params.push_back(nlogthresholds);
    expandParameters(params, gridExpandedParameters);
    
    if (m_bModelSelection)
    {
		cout << m_data.getModality() << " model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " ";
            
            // Index the k-th training
            GridMat descriptorsTrainGrid (descriptorsGrid, partitionsGrid, k, true);
            GridMat validnessesTrainGrid (validnessesGrid, partitionsGrid, k, true);
            GridMat tagsTrainGrid (tagsGrid, partitionsGrid, k, true);
            
//            // <-- dbg
//            cv::Mat d = descriptorsTrainGrid.at(0,0);
//            cv::Mat v = validnessesTrainGrid.at(0,0);
//            cv::Mat t = tagsTrainGrid.at(0,0);
//            for (int i = 0; i < d.rows; i++)
//            {
//                cout << d.row(i) << " " << v.row(i) << " " << t.row(i) << endl;
//            }
//           // dgb -->
            
            // Within the k-th training partition,
            // remove the nonvalid descriptors (validness == 0) and associated tags
            GridMat validDescriptorsTrainGrid = descriptorsTrainGrid.convertToDense(validnessesTrainGrid);
            GridMat validTagsTrainGrid = tagsTrainGrid.convertToDense(validnessesTrainGrid);
            
            if (m_bTrainMirrored)
            {
                GridMat descriptorsMirroredTrainGrid (m_data.getDescriptorsMirrored(), partitionsGrid, k, true);
                GridMat validnessesMirroredTrainGrid (m_data.getValidnessesMirrored(), partitionsGrid, k, true);
                
                GridMat validDescriptorsMirroredTrainGrid = descriptorsMirroredTrainGrid.convertToDense(validnessesMirroredTrainGrid);
                GridMat validTagsMirroredTrainGrid = tagsTrainGrid.convertToDense(validnessesMirroredTrainGrid);
                
                validDescriptorsTrainGrid.vconcat(validDescriptorsMirroredTrainGrid);
                validTagsTrainGrid.vconcat(validTagsMirroredTrainGrid);
            }
            
            GridMat goodnesses;
            modelSelection(validDescriptorsTrainGrid, validTagsTrainGrid,
                           gridExpandedParameters, goodnesses);

            std::stringstream ss;
            ss << m_data.getModality() << "_models_goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            goodnesses.save(ss.str());
        }
        cout << endl;
    }
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    GridMat goodnesses (m_data.getHp(), m_data.getWp());
    for (int k = 0; k < m_testK; k++)
    {
        // Model selection information is kept on disk, reload it
        GridMat aux;
        std::stringstream ss;
        ss << m_data.getModality() << "_models_goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        aux.load(ss.str());
        goodnesses.hconcat(aux);
    }
    
    m_PredictionsGrid.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<int>::type));
    m_LoglikelihoodsGrid.setTo(cv::Mat(tags.rows, tags.cols, cv::DataType<float>::type, cv::Scalar(std::numeric_limits<float>::min())));
    m_DistsToMarginGrid.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<float>::type));
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        // Index the k-th training and test partitions
        GridMat descriptorsTrainGrid (descriptorsGrid, partitionsGrid, k, true);
        GridMat descriptorsTestGrid (descriptorsGrid, partitionsGrid, k);
        
        GridMat validnessesTrainGrid (validnessesGrid, partitionsGrid, k, true);
        GridMat validnessesTestGrid (validnessesGrid, partitionsGrid, k);
        
        GridMat tagsTrainGrid (tagsGrid, partitionsGrid, k, true);
        GridMat tagsTestGrid (tagsGrid, partitionsGrid, k);
        
        // Within the k-th training partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTrainGrid = descriptorsTrainGrid.convertToDense(validnessesTrainGrid);
        GridMat validTagsTrainGrid = tagsTrainGrid.convertToDense(validnessesTrainGrid);
                
        // Within the valid descriptors in the k-th training partition,
        // index the subject descriptors (tag == 1)
        GridMat validSbjDescriptorsTrainGrid (validDescriptorsTrainGrid, validTagsTrainGrid, 1);
        
        GridPredictor<cv::EM40> predictor(m_hp, m_wp);
       
//        // Model selection information is kept on disk, reload it
//        GridMat goodnesses;
//        std::stringstream ss;
//        ss << m_data.getModality() << "_models_goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
//        goodnesses.load(ss.str());
        
        GridMat goodness;
        if (m_bGlobalBest)
        {
            GridMat globalMean;
            goodnesses.mean(goodness, 1);
        }
        else
        {
            goodness = goodnesses.col(k);
        }
        
        // Train with the best parameter combination in average in a model
        // selection procedure within the training partition
        vector<cv::Mat> bestParams;
        selectBestParameterCombination<float>(gridExpandedParameters, m_hp, m_wp, params.size(), goodness, bestParams);

        predictor.setNumOfMixtures(bestParams[0]);
        predictor.setEpsilons(bestParams[1]);
        predictor.setLoglikelihoodThreshold(bestParams[2]);
        
        if (m_bDimReduction)
            predictor.setDimensionalityReduction(cv::Mat(m_hp, m_wp, cv::DataType<double>::type, cv::Scalar(m_variance)));

        // Training phase
        if (m_bTrainMirrored)
        {
            GridMat descriptorsMirroredTrainGrid (m_data.getDescriptorsMirrored(), partitionsGrid, k, true);
            GridMat validnessesMirroredTrainGrid (m_data.getValidnessesMirrored(), partitionsGrid, k, true);
            
            GridMat validDescriptorsMirroredTrainGrid = descriptorsMirroredTrainGrid.convertToDense(validnessesMirroredTrainGrid);
            GridMat validTagsMirroredTrainGrid = tagsTrainGrid.convertToDense(validnessesMirroredTrainGrid);
            GridMat validSbjDescriptorsMirroredTrainGrid (validDescriptorsMirroredTrainGrid, validTagsMirroredTrainGrid, 1);
            
            validSbjDescriptorsTrainGrid.vconcat(validSbjDescriptorsMirroredTrainGrid);
        }
        
        predictor.train(validSbjDescriptorsTrainGrid);
        
        // Predict phase
        
        // Within the k-th test partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validDescriptorsTestGrid = descriptorsTestGrid.convertToDense(validnessesTestGrid);
        
        GridMat validPredictionsGrid, validLoglikelihoodsGrid, validDistsToMarginGrid;
        predictor.predict(validDescriptorsTestGrid, validPredictionsGrid, validLoglikelihoodsGrid, validDistsToMarginGrid);
        
//        GridMat validTagsTestGrid = tagsTestGrid.convertToDense(validnessesTestGrid);
//        GridMat validTagsSbjObjTeFold (validtagsTestGrid, validTagsTestGrid, -1, true);
//        GridMat validPredictionsSbjObjTeFold (validPredictions, validTagsTestGrid, -1, true);

        // Store other results apart from the goodness
        m_PredictionsGrid.set(validPredictionsGrid.convertToSparse(validnessesTestGrid), partitionsGrid, k);
        m_LoglikelihoodsGrid.set(validLoglikelihoodsGrid.convertToSparse(validnessesTestGrid), partitionsGrid, k);
        m_DistsToMarginGrid.set(validDistsToMarginGrid.convertToSparse(validnessesTestGrid), partitionsGrid, k);
    }
    cout << endl;

    predictionsGrid = m_PredictionsGrid; // TODO: debug predictions output, wheter [0,255] or not..
    loglikelihoodsGrid = m_LoglikelihoodsGrid;
    distsToMarginGrid = m_DistsToMarginGrid;
}

template<typename T>
//void ModalityPrediction<cv::EM40>::modelSelection(cv::Mat descriptors, cv::Mat tags,
//                                                vector<vector<T> > gridExpandedParams,
//                                                cv::Mat& goodness)
void ModalityPrediction<cv::EM40>::modelSelection(GridMat descriptors, GridMat tags,
                                                  vector<vector<T> > gridExpandedParams,
                                                  GridMat& goodnesses)
{
    GridMat partitions;
    cvpartition(tags, m_modelSelecK, m_seed, partitions);
    
    GridMat accuracies;
    accuracies.setTo(cv::Mat(gridExpandedParams.size(), m_modelSelecK, cv::DataType<float>::type));
    
    GridMat descriptorsaux = descriptors;
    
    boost::timer t;
    boost::thread_group tg;

    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        // Get fold's data
        cout << k;
    
        GridMat descriptorsTrainGrid (descriptors, partitions, k, true);
        GridMat descriptorsValidationGrid (descriptors, partitions, k);
        
        GridMat tagsTrainGrid (tags, partitions, k, true);
        GridMat tagsValidationGrid (tags, partitions, k);
        
        GridMat descriptorsSbjTrainGrid (descriptorsTrainGrid, tagsTrainGrid, 1); // subjects' training sample
        GridMat descriptorsSbjObjValidationGrid (descriptorsValidationGrid, tagsValidationGrid, -1, true);
        
        GridMat tagsSbjObjValidationGrid (tagsValidationGrid, tagsValidationGrid, -1, true);
        
//        boost::bind(&ModalityPrediction::_modelSelection<float>, this, _1, _2, _3, _4, _5, _6)(descriptorsSbjTrainGrid, descriptorsSbjObjValidationGrid, tagsSbjObjValidationGrid, k, gridExpandedParams, boost::ref(accuracies));
        tg.add_thread(new boost::thread( boost::bind (&ModalityPrediction::_modelSelection<T>, this,descriptorsSbjTrainGrid, descriptorsSbjObjValidationGrid, tagsSbjObjValidationGrid, k, gridExpandedParams, accuracies) ));
    }
    tg.join_all();
    
    cout << ") " << t.elapsed() << endl;

    accuracies.mean(goodnesses, 1);
}

template<typename T>
void ModalityPrediction<cv::EM40>::_modelSelection(GridMat descriptorsSbjTrainGrid, GridMat descriptorsSbjObjValGrid, GridMat tagsSbjObjValGrid, int k, vector<vector<T> > gridExpandedParams, GridMat& accs)
{
    GridMat accsFold; // results

    for (int i = 0; i < m_data.getHp(); i++) for (int j = 0; j < m_data.getWp(); j++)
    {
        cv::PCA pca; // if m_bDimReduction is false, this variable is not used anymore
        
        cv::Mat descriptorsSbjTrain;
        if (!m_bDimReduction)
            descriptorsSbjTrain = descriptorsSbjTrainGrid.at(i,j);
        else
            cvx::computePCA(descriptorsSbjTrainGrid.at(i,j), pca,
                            descriptorsSbjTrain, CV_PCA_DATA_AS_ROW, m_variance);
        
        cv::EM40 predictor;
        for (int m = 0; m < gridExpandedParams.size(); m++)
        {
            vector<T> combination = gridExpandedParams[m];
            // Create predictor and its parametrization
            int nclusters = predictor.get<int>("nclusters");
            float epsilon = predictor.get<float>("epsilon");
            if (combination[0] != nclusters || combination[1] != epsilon)
            {
                predictor.clear();
                predictor.set("nclusters", combination[0]);
                predictor.set("epsilon", combination[1]);
                
                // Train
                predictor.train(descriptorsSbjTrain);
            }
            
            // Test
            cv::Mat_<int> labels;
            cv::Mat_<float> loglikelihoods;
            for (int d = 0; d < descriptorsSbjObjValGrid.at(i,j).rows; d++)
            {
                cv::Mat descriptor = descriptorsSbjObjValGrid.at(i,j).row(d);
                
                if (m_bDimReduction)
                    descriptor = pca.project(descriptor);
                
                cv::Vec3d res = predictor.predict(descriptor);
                
                labels.push_back(res.val[2]);
                loglikelihoods.push_back(res.val[1]);
            }
            
            // Standardized loglikelihoods
            cv::Mat_<float> means, stddevs;
            means.create(loglikelihoods.rows, loglikelihoods.cols);
            stddevs.create(loglikelihoods.rows, loglikelihoods.cols);
            for (int l = 0; l < nclusters; l++)
            {
                cv::Scalar mean, stddev;
                cv::meanStdDev(loglikelihoods, mean, stddev, labels == l);
                means.setTo(mean.val[0], labels == l);
                stddevs.setTo(stddev.val[0], labels == l);
            }
            cv::Mat_<float> ctrLoglikelihoods, stdLoglikelihoods;
            cv::subtract(loglikelihoods, means, ctrLoglikelihoods);
            cv::divide(ctrLoglikelihoods, stddevs, stdLoglikelihoods);
            
            // Predictions evaluation comparing the standardized loglikelihoods to a threshold,
            // loglikelihoods over threshold are considered subject (1)
            cv::Mat predictions;
            cv::threshold(stdLoglikelihoods, predictions, gridExpandedParams[m][2], 1, CV_THRESH_BINARY);
            predictions.convertTo(predictions, cv::DataType<int>::type);
            

            
            // Compute an accuracy measure
            float acc = accuracy(tagsSbjObjValGrid.at(i,j), predictions);
            accsFold.at(i,j).push_back(acc); // element to row
        }
        
        m_mutex.lock();
        accsFold.at(i,j).copyTo(accs.at(i,j).col(k));
        m_mutex.unlock();
    }
}

void ModalityPrediction<cv::EM40>::computeLoglikelihoodsDistribution(int nbins, double min, double max, cv::Mat& sbjDistribution, cv::Mat& objDistribution)
{
    cv::Mat tags = m_data.getTagsMat();
    GridMat gtags;
    gtags.setTo(tags);
    
    GridMat descriptorsGrid = m_data.getDescriptors();
    GridMat gvalidnesses = m_data.getValidnesses();
    
    GridMat partitionsGrid;
    cvpartition(gtags, m_testK, m_seed, partitionsGrid);
    
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
        GridMat descriptorsTrainGrid (descriptorsGrid, partitionsGrid, k, true);
        GridMat descriptorsTeFold (descriptorsGrid, partitionsGrid, k);
        GridMat validnessesTrainGrid (gvalidnesses, partitionsGrid, k, true);
        GridMat validnessesTeFold (gvalidnesses, partitionsGrid, k);
        GridMat tagsTrainGrid (gtags, partitionsGrid, k, true);
        GridMat tagsTestGrid (gtags, partitionsGrid, k);
        
        // Within the k-th training partition,
        // remove the nonvalid descriptors (validness == 0) and associated tags
        GridMat validdescriptorsTrainGrid = descriptorsTrainGrid.convertToDense(validnessesTrainGrid);
        GridMat validtagsTrainGrid = tagsTrainGrid.convertToDense(validnessesTrainGrid);
        
        GridMat validDescriptorsTeFold = descriptorsTeFold.convertToDense(validnessesTeFold);
        GridMat validtagsTestGrid = tagsTestGrid.convertToDense(validnessesTeFold);
        
        // Within the valid descriptors in the k-th training partition,
        // index the subject descriptors (tag == 1)
        GridMat validSbjdescriptorsTrainGrid (validdescriptorsTrainGrid, validtagsTrainGrid, 1);
        
        GridMat validObjDescriptorsTeFold (validDescriptorsTeFold, validtagsTestGrid, 0);
        GridMat validSbjDescriptorsTeFold (validDescriptorsTeFold, validtagsTestGrid, 1);
        
        GridPredictor<cv::EM40> predictor(m_hp, m_wp);
        
        for (int m = 0; m < gridExpandedParams.size(); m++)
        {
//            cout << m << " ";
            
            cv::Mat numOfMixtures;
            numOfMixtures.setTo(gridExpandedParams[m][0]);
            predictor.setNumOfMixtures(numOfMixtures);
            
            // Training phase
            
            predictor.train(validSbjdescriptorsTrainGrid);
            
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

void ModalityPrediction<cv::Mat>::predict(GridMat& predictionsGrid, GridMat& ramananScoresGrid, GridMat& distsToMarginGrid)
{
    cv::Mat tags = m_data.getTagsMat();
    
    GridMat gtags;
    gtags.setTo(tags);
    GridMat gvalidnesses = m_data.getValidnesses();
    
    cv::Mat partitions = m_data.getPartitions();
//    GridMat partitionsGrid (partitions);
    
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
    
    m_PredictionsGrid.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<int>::type));
    m_RamananScoresGrid.setTo(cv::Mat(tags.rows, tags.cols, cv::DataType<float>::type, cv::Scalar(std::numeric_limits<float>::min())));
    m_DistsToMarginGrid.setTo(cv::Mat::zeros(tags.rows, tags.cols, cv::DataType<float>::type));
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        // Index the k-th test partitions
        cv::Mat indicesTeFold = cvx::indexMat(cvx::linspace(0, tags.rows), partitions == k);
//        GridMat validnessesTeFold (gvalidnesses, partitionsGrid, k);
//        GridMat tagsTestGrid (gtags, partitionsGrid, k);
        cv::Mat tagsTestGrid = cvx::indexMat(tags, partitions == k);
        
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
            cv::Mat validTagsTe = cvx::indexMat(tagsTestGrid, validnessesTe);
        
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
            
            cvx::setMat(validPredictions, predictions, validnessesTe); // not indexed by validnessesTe take 0
            cvx::setMat(validScores, scores, validnessesTe); // not indexed by validnessesTe take valids' mean
            cvx::setMat(stdValidDistances, distances, validnessesTe); // not indexed by validnessesTe take 0
            
            cvx::setMat(predictions, m_PredictionsGrid.at(i,j), partitions == k);
            cvx::setMat(scores, m_RamananScoresGrid.at(i,j), partitions == k);
            cvx::setMat(distances, m_DistsToMarginGrid.at(i,j), partitions == k);
        }
    }
    cout << endl;
    
    predictionsGrid = m_PredictionsGrid;
    ramananScoresGrid = m_RamananScoresGrid;
    distsToMarginGrid = m_DistsToMarginGrid;
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
template void ModalityPredictionBase<cv::EM40>::setData(ModalityGridData &data);
template void ModalityPredictionBase<cv::EM40>::setPredictions(GridMat predictionsGrid);
template void ModalityPredictionBase<cv::EM40>::setDistsToMargin(GridMat distsToMarginGrid);
template void ModalityPredictionBase<cv::EM40>::setModelSelection(bool flag);
template void ModalityPredictionBase<cv::EM40>::setModelSelectionParameters(int k, bool best);
template void ModalityPredictionBase<cv::EM40>::setValidationParameters(int k);
template void ModalityPredictionBase<cv::EM40>::setDimensionalityReduction(float variance);
template void ModalityPredictionBase<cv::EM40>::setTrainMirrored(bool flag);
template void ModalityPredictionBase<cv::EM40>::getAccuracy(cv::Mat predictions, cv::Mat &accuracies);
template void ModalityPredictionBase<cv::EM40>::getAccuracy(GridMat predictions, GridMat &accuracies);
template void ModalityPredictionBase<cv::EM40>::computeGridConsensusPredictions(cv::Mat& consensusPredictions,
                                                                              cv::Mat& consensusDistsToMargin);
template void ModalityPrediction<cv::EM40>::modelSelection<int>(GridMat descriptors, GridMat tags, vector<vector<int> > params, GridMat& goodness);
template void ModalityPrediction<cv::EM40>::modelSelection<float>(GridMat descriptors, GridMat tags, vector<vector<float> > params, GridMat& goodness);
template void ModalityPrediction<cv::EM40>::modelSelection<double>(GridMat descriptors, GridMat tags, vector<vector<double> > params, GridMat& goodness);

template void ModalityPrediction<cv::EM40>::_modelSelection<int>(GridMat descriptorsSbjTrainGrid, GridMat descriptorsSbjObjValGrid, GridMat tagsSbjObjValGrid, int k, vector<vector<int> > gridExpandedParams, GridMat& accs);
template void ModalityPrediction<cv::EM40>::_modelSelection<float>(GridMat descriptorsSbjTrainGrid, GridMat descriptorsSbjObjValGrid, GridMat tagsSbjObjValGrid, int k, vector<vector<float> > gridExpandedParams, GridMat& accs);
template void ModalityPrediction<cv::EM40>::_modelSelection<double>(GridMat descriptorsSbjTrainGrid, GridMat descriptorsSbjObjValGrid, GridMat tagsSbjObjValGrid, int k, vector<vector<double> > gridExpandedParams, GridMat& accs);

template void ModalityPredictionBase<cv::Mat>::setData(ModalityGridData &data);
template void ModalityPredictionBase<cv::Mat>::setPredictions(GridMat predictionsGrid);
template void ModalityPredictionBase<cv::Mat>::setDistsToMargin(GridMat distsToMarginGrid);
template void ModalityPredictionBase<cv::Mat>::setModelSelection(bool flag);
template void ModalityPredictionBase<cv::Mat>::setModelSelectionParameters(int k, bool best);
template void ModalityPredictionBase<cv::Mat>::setValidationParameters(int k);
template void ModalityPredictionBase<cv::Mat>::getAccuracy(cv::Mat predictions, cv::Mat &accuracies);
template void ModalityPredictionBase<cv::Mat>::getAccuracy(GridMat predictions, GridMat &accuracies);
template void ModalityPredictionBase<cv::Mat>::computeGridConsensusPredictions(cv::Mat& consensusPredictions,
                                                                               cv::Mat& consensusDistsToMargin);
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
