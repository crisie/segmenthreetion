//
//  FusionPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#include "FusionPrediction.h"
#include "StatTools.h"
#include <boost/assign/std/vector.hpp>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

using namespace boost::assign;

// Instantiation of template member functions
// -----------------------------------------------------------------------------

template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
//template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setModelSelectionParameters(int, int, bool);
template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setTrainMirrored(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setValidationParameters(int);
template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setStackedPrediction(bool flag);
template cv::Mat ClassifierFusionPredictionBase<cv::EM40,CvBoost>::getAccuracies();
//template void ClassifierFusionPredictionBase<cv::EM40,CvBoost>::setPartitions(cv::Mat partitions);


template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
//template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setModelSelectionParameters(int, int, bool);
template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setTrainMirrored(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setValidationParameters(int);
template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setStackedPrediction(bool flag);
template cv::Mat ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::getAccuracies();
//template void ClassifierFusionPredictionBase<cv::EM40,CvANN_MLP>::setPartitions(cv::Mat partitions);


template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin, vector<cv::Mat> predictions);
//template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setModelSelectionParameters(int, int, bool);
template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setTrainMirrored(bool flag);
template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setValidationParameters(int);
template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setStackedPrediction(bool flag);
template cv::Mat ClassifierFusionPredictionBase<cv::EM40,CvSVM>::getAccuracies();
//template void ClassifierFusionPredictionBase<cv::EM40,CvSVM>::setPartitions(cv::Mat partitions);

// -----------------------------------------------------------------------------


SimpleFusionPrediction::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction::setModalitiesData(vector<ModalityGridData> mgds)
{
    m_mgds = mgds;
    m_partitions = m_mgds[0].getPartitions();
    m_tags = m_mgds[0].getTagsMat();
    for (int i = 0; i < m_mgds.size(); i++)
    {
        // partitions' values among modalities coincide
        assert ( cv::sum(m_partitions == m_mgds[i].getPartitions()).val[0] / 255 == m_partitions.rows );
        // same check for tags
        assert ( cv::sum(m_tags == m_mgds[i].getTagsMat()).val[0] / 255 == m_tags.rows );
    }
}

void SimpleFusionPrediction::predict(vector<cv::Mat> allPredictions, vector<cv::Mat> allDistsToMargin,
                                     cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    cv::Mat predictions = allPredictions[0];
    cv::Mat distsToMargin = allDistsToMargin[0];
    
    for (int i = 1; i < allPredictions.size(); i++)
    {
        cv::hconcat(predictions,   allPredictions[i],   predictions);
        cv::hconcat(distsToMargin, allDistsToMargin[i], distsToMargin);
    }
    
    int n = predictions.rows;
    
    fusionPredictions.create(predictions.rows, 1, predictions.type());
    fusionDistsToMargin.create(distsToMargin.rows, 1, distsToMargin.type());
    
    for (int k = 0; k < n; k++)
    {
        int pos = 0;
        int neg = 0;
        float accPosDists = 0;
        float accNegDists = 0;
        for (int m = 0; m < predictions.cols; m++)
        {
            float dist = distsToMargin.at<float>(k,m);
            if (predictions.at<int>(k,m) == 0)
            {
                neg++;
                accNegDists += dist;
            }
            else
            {
                pos++;
                accPosDists += dist;
            }
        }
        
        if (pos < neg)
        {
            fusionPredictions.at<int>(k,0) = 0;
            fusionDistsToMargin.at<float>(k,0) = accNegDists/neg;
        }
        else if (pos > neg)
        {
            fusionPredictions.at<int>(k,0) = 1;
            fusionDistsToMargin.at<float>(k,0) = accPosDists/pos;
        }
        else
        {
            if (accPosDists/pos < abs(accNegDists/neg))
            {
                fusionPredictions.at<int>(k,0) = 0;
                fusionDistsToMargin.at<float>(k,0) = accNegDists/neg;
            }
            else
            {
                fusionPredictions.at<int>(k,0) = 1;
                fusionDistsToMargin.at<float>(k,0) = accPosDists/pos;
            }
                
        }
    }
    
    m_fusionPredictions = fusionPredictions;
    m_fusionDistsToMargin = fusionDistsToMargin;
}

void SimpleFusionPrediction::predict(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin)
{
    GridMat fusionPredictionsGrid, fusionDistsToMarginGrid;
    
    int hp = allPredictions[0].crows();
    int wp = allPredictions[0].ccols();
    for (int m = 1; m < allPredictions.size(); m++)
    {
        assert (allPredictions[m].crows() == hp || allPredictions[m].ccols() == wp);
        assert (allDistsToMargin[m].crows() == hp || allDistsToMargin[m].ccols() == wp);
    }
    
    fusionPredictionsGrid.create(hp, wp);
    fusionDistsToMarginGrid.create(hp, wp);
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        vector<cv::Mat> predictions, distsToMargin;
        for (int m = 0; m < allPredictions.size(); m++)
        {
            predictions += allPredictions[m].at(i,j);
            distsToMargin += allDistsToMargin[m].at(i,j);
        }

        predict(predictions, distsToMargin, fusionPredictionsGrid.at(i,j), fusionDistsToMarginGrid.at(i,j));
    }
    
    computeGridConsensusPredictions(fusionPredictionsGrid, fusionDistsToMarginGrid, fusionPredictions, fusionDistsToMargin);
    
    m_fusionPredictions = fusionPredictions;
    m_fusionDistsToMargin = fusionDistsToMargin;
}

void SimpleFusionPrediction::predict(vector<GridMat> allDistsToMargin, cv::Mat& fusionPredictions, cv::Mat& fusionDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat concatDistsToMarginGrid;
    
    for (int i = 0; i < allDistsToMargin.size(); i++)
    {
        concatDistsToMarginGrid.hconcat(allDistsToMargin[i]);
    }
    
    GridMat fusionPredictionsGrid, fusionDistsToMarginGrid;
    predict(concatDistsToMarginGrid, fusionPredictionsGrid, fusionDistsToMarginGrid);
    
    computeGridConsensusPredictions(fusionPredictionsGrid, fusionDistsToMarginGrid, fusionPredictions, fusionDistsToMargin);
    
    m_fusionPredictions = fusionPredictions;
    m_fusionDistsToMargin = fusionDistsToMargin;
}

void SimpleFusionPrediction::predict(GridMat distsToMarginGrid, GridMat& fusionPredictionsGrid, GridMat& fusionDistsToMarginGrid)
{
    int hp = distsToMarginGrid.crows();
    int wp = distsToMarginGrid.ccols();
    int n = distsToMarginGrid.at(0,0).rows;
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        assert (n == distsToMarginGrid.at(i,j).rows);
    }
    
    fusionPredictionsGrid.create<int>(hp, wp, n, 1);
    fusionDistsToMarginGrid.create<float>(hp, wp, n, 1);
    
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            int pos = 0;
            int neg = 0;
            float accPosDists = 0;
            float accNegDists = 0;
            for (int m = 0; m < distsToMarginGrid.at(i,j).cols; m++)
            {
                float dist = distsToMarginGrid.at<float>(i,j,k,m);
                if (distsToMarginGrid.at<int>(i,j,k,m) < 0)
                {
                    neg++;
                    accNegDists += dist;
                }
                else
                {
                    pos++;
                    accPosDists += dist;
                }
            }
            
            if (pos < neg)
            {
                fusionPredictionsGrid.at<int>(i,j,k,0) = 0;
                fusionDistsToMarginGrid.at<float>(i,j,k,0) = accNegDists/neg;
            }
            else if (pos > neg)
            {
                fusionPredictionsGrid.at<int>(i,j,k,0) = 1;
                fusionDistsToMarginGrid.at<float>(i,j,k,0) = accPosDists/pos;
            }
            else
            {
                float meanNeg = accNegDists/neg;
                float meanPos = accPosDists/pos;
                if (meanPos > abs(meanNeg))
                {
                    fusionPredictionsGrid.at<int>(i,j,k,0) = 1;
                    fusionDistsToMarginGrid.at<float>(i,j,k,0) = accPosDists/pos;
                }
                else
                {
                    fusionPredictionsGrid.at<int>(i,j,k,0) = 0;
                    fusionDistsToMarginGrid.at<float>(i,j,k,0) = accNegDists/neg;
                }
            }
        }
        
    }
}

void SimpleFusionPrediction::computeGridConsensusPredictions(GridMat fusionPredictionsGrid,
                                                             GridMat fusionDistsToMarginGrid,
                                                             cv::Mat& consensusfusionPredictions,
                                                             cv::Mat& consensusfusionDistsToMargin)
{
    consensusfusionPredictions.create(m_tags.rows, 1, cv::DataType<int>::type);
    consensusfusionDistsToMargin.create(m_tags.rows, 1, cv::DataType<float>::type);
    
    //    cv::Mat partitions;
    //    cvpartition(tags, m_testK, m_seed, partitions);
    
    for (int r = 0; r < m_tags.rows; r++)
    {
        int pos = 0;
        int neg = 0;
        float accPosDists = 0;
        float accNegDists = 0;
        for (int i = 0; i < fusionPredictionsGrid.crows(); i++) for (int j = 0; j < fusionPredictionsGrid.ccols(); j++)
        {
            if (fusionPredictionsGrid.at<int>(i,j,r,0) == 0)
            {
                neg++;
                accNegDists += fusionDistsToMarginGrid.at<float>(i,j,r,0);
            }
            else if (fusionPredictionsGrid.at<int>(i,j,r,0) == 1)
            {
                pos++;
                accPosDists += fusionDistsToMarginGrid.at<float>(i,j,r,0);
            }
        }
        
        if (pos > neg)
        {
            consensusfusionPredictions.at<int>(r,0) = 1;
            consensusfusionDistsToMargin.at<float>(r,0) = accPosDists / pos;
        }
        else if (pos < neg)
        {
            consensusfusionPredictions.at<int>(r,0) = 0;
            consensusfusionDistsToMargin.at<float>(r,0) = accNegDists / neg;
        }
        else // pos == neg
        {
            if (accPosDists > abs(accNegDists)) // most confident towards positive classification
            {
                consensusfusionPredictions.at<int>(r,0) = 1;
                consensusfusionDistsToMargin.at<float>(r,0) = accPosDists / pos;
            }
            else
            {
                consensusfusionPredictions.at<int>(r,0) = 0;
                consensusfusionDistsToMargin.at<float>(r,0) = accNegDists / neg;
            }
        }
    }
}

cv::Mat SimpleFusionPrediction::getAccuracies()
{
    cv::Mat accuracies;
    
    accuracy(m_tags, m_fusionPredictions, m_partitions, accuracies);
    
    return accuracies;
}

//
// ClassifierFusionPredictionBase class
//

template<typename ClassifierT>
ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::ClassifierFusionPredictionBase()
: m_pClassifier(new ClassifierT), m_bModelSelection(true), m_bStackPredictions(false), m_narrowSearchSteps(7)
{
    
}

//template<typename ClassifierT>
//void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin)
//{
//    m_mgds = mgds;
//    m_distsToMargin = distsToMargin;
//}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setData(vector<ModalityGridData> mgds, vector<GridMat> distsToMargin, vector<cv::Mat> predictions)
{
    m_mgds = mgds;
    m_partitions = m_mgds[0].getPartitions();
    m_responses = m_mgds[0].getTagsMat();
    for (int i = 0; i < m_mgds.size(); i++)
    {
        // partitions' values among modalities coincide
        assert ( cv::sum(m_partitions == m_mgds[i].getPartitions()).val[0] / 255 == m_partitions.rows );
        // same check for tags
        assert ( cv::sum(m_responses == m_mgds[i].getTagsMat()).val[0] / 255 == m_responses.rows );
    }
    
    m_distsToMargin = distsToMargin;
    m_predictions = predictions;
}

//template<typename ClassifierT>
//void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setResponses(cv::Mat responses)
//{
//    m_responses = responses;
//}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setModelSelection(bool flag)
{
    m_bModelSelection = flag;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setModelSelectionParameters(int k, int seed, bool bGlobalBest)
{
    m_modelSelecK = k;
    m_seed = seed;
    m_bGlobalBest = bGlobalBest;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setValidationParameters(int k)
{
    m_testK = k;
}

//template<typename ClassifierT>
//void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::setPartitions(cv::Mat partitions)
//{
//    m_partitions = partitions;
//}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40, ClassifierT>::formatData()
{
    m_data.release();
    m_data.create(m_distsToMargin[0].at(0,0).rows, 0, m_distsToMargin[0].at(0,0).type());
    
    // Build here a data structure needed to feed the classifier
    // Better use GridMat functions...
    for (int i = 0; i < m_distsToMargin.size(); i++)
    {
        cv::Mat serialMat;
        m_distsToMargin[i].hserial(serialMat);
        
        if (m_data.cols == 0)
            m_data = serialMat;
        else
            cv::hconcat(m_data, serialMat, m_data);
        
        if (m_bStackPredictions)
        {
            cv::Mat_<float> normPredictions = 2 * m_predictions[i] - 1;
            cv::hconcat(m_data, normPredictions, m_data);
        }
    }
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40,ClassifierT>::setTrainMirrored(bool flag)
{
    m_bTrainMirrored = flag;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM40,ClassifierT>::setStackedPrediction(bool flag)
{
    m_bStackPredictions = flag;
}

template<typename ClassifierT>
cv::Mat ClassifierFusionPredictionBase<cv::EM40,ClassifierT>::getAccuracies()
{
    cv::Mat accuracies;
    
    accuracy(m_responses, m_fusionPredictions, m_partitions, accuracies);
    
    return accuracies;
}

//
// ClassifierFusionPrediction class templates' specialization
//

// SVM

ClassifierFusionPrediction<cv::EM40,CvSVM>::ClassifierFusionPrediction()
: m_numItersSVM(10000)
{
    
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::setKernelType(int type)
{
    m_kernelType = type;
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::setCs(vector<float> cs)
{
    m_cs = cs;
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::setGammas(vector<float> gammas)
{
    m_gammas = gammas;
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::predict(cv::Mat& fusionPredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_cs);
    if (m_kernelType == CvSVM::RBF)
        params.push_back(m_gammas);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
//    cv::Mat partitions;
//    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Coarse model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            std::stringstream coarsess;
            coarsess << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat coarseGoodnesses, aux;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (coarseGoodnesses.empty()) coarseGoodnesses = aux.col(params.size());
        else cv::hconcat(coarseGoodnesses, aux.col(params.size()), coarseGoodnesses);
    }
    cv::reduce(coarseGoodnesses, aux, 1, CV_REDUCE_AVG);
    cv::hconcat(coarseExpandedParameters, aux, coarseGoodnesses);
    
    cv::Mat narrowExpandedParameters;
    int discretes[] = {0,0};
    narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
    
    if (m_bModelSelection)
    {
        cout << "Narrow model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
            
            // Narrow search
            
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat goodnesses;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "svm_" << m_distsToMargin.size() << "_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (goodnesses.empty()) goodnesses = aux.col(params.size());
        else cv::hconcat(goodnesses, aux.col(params.size()), goodnesses);
    }
//    cv::reduce(goodnesses, aux, 1, CV_REDUCE_AVG);
//    cv::hconcat(coarseExpandedParameters, aux, goodnesses);
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    fusionPredictions.create(m_responses.rows, 1, cv::DataType<int>::type);
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodness;
        if (m_bGlobalBest)
        {
            GridMat globalMean;
            cv::reduce(goodnesses, goodness, 1, CV_REDUCE_AVG);
        }
        else
        {
            goodness = goodnesses.col(k);
        }
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodness, &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestC     = narrowExpandedParameters.row(best.y).at<float>(0,0);
        float bestGamma = narrowExpandedParameters.row(best.y).at<float>(0,1);
        
        CvSVMParams svmParams (CvSVM::C_SVC, m_kernelType, 0, bestGamma, 0, bestC, 0, 0, 0,
                               cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, 1e-2 ));
        m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(), svmParams);
        
        // Prediction phase
        cv::Mat tePredictions;
        m_pClassifier->predict(teData, tePredictions);
        
        cvx::setMat(tePredictions, fusionPredictions, m_partitions == k);
    }
    cout << endl;
    
    m_fusionPredictions = fusionPredictions;
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::_modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies)
{
    cv::Mat foldAccs (expandedParams.rows, 1, cv::DataType<float>::type); // results
    
    for (int m = 0; m < expandedParams.rows; m++)
    {
        // Training phase
        cv::Mat selectedParams = expandedParams.row(m);
        
        float C = selectedParams.at<float>(0,0);
        
        float gamma = 0;
        if (m_kernelType == CvSVM::RBF)
            gamma = selectedParams.at<float>(0,1); // indeed, gamma not used if not RBF kernel
        
        CvSVM classifier;
        CvSVMParams params (CvSVM::C_SVC, m_kernelType, 0, gamma, 0, C, 0, 0, 0,
                            cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, 1e-2 ));
        
        classifier.train(descriptorsTr, responsesTr, cv::Mat(), cv::Mat(), params);
        
        // Test phase
        cv::Mat predictionsVal;
        classifier.predict(descriptorsVal, predictionsVal);
        
        // Compute an accuracy measure
        foldAccs.at<float>(m,0) = accuracy(responsesVal, predictionsVal);
    }
    
    m_mutex.lock();
    foldAccs.copyTo(accuracies.col(k));
    m_mutex.unlock();
}

void ClassifierFusionPrediction<cv::EM40,CvSVM>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, m_modelSelecK, cv::DataType<float>::type);
    
    boost::thread_group tg;
    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k;
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        tg.add_thread(new boost::thread( boost::bind(&ClassifierFusionPrediction::_modelSelection, this, trSbjObjData, trSbjObjResponses, valData, valResponses, k, expandedParams, accuracies) ));
    }
    tg.join_all();
    cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}


// CvBoost

ClassifierFusionPrediction<cv::EM40,CvBoost>::ClassifierFusionPrediction()
{
    
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::setBoostType(int type)
{
    m_boostType = type;
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::setNumOfWeaks(vector<float> numOfWeaks)
{
    m_numOfWeaks = numOfWeaks;
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::setWeightTrimRate(vector<float> weightTrimRates)
{
    m_weightTrimRate = weightTrimRates;
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::predict(cv::Mat& fusionPredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_numOfWeaks);
    if (m_boostType == CvBoost::GENTLE)
        params.push_back(m_weightTrimRate);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
//    cv::Mat partitions;
//    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Coarse model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            std::stringstream coarsess;
            coarsess << "boost_" << m_distsToMargin.size() << "_" << m_boostType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat coarseGoodnesses, aux;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "boost_" << m_distsToMargin.size() << "_" << m_boostType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (coarseGoodnesses.empty()) coarseGoodnesses = aux.col(params.size());
        else cv::hconcat(coarseGoodnesses, aux.col(params.size()), coarseGoodnesses);
    }
    cv::reduce(coarseGoodnesses, aux, 1, CV_REDUCE_AVG);
    cv::hconcat(coarseExpandedParameters, aux, coarseGoodnesses);
    
    cv::Mat narrowExpandedParameters;
    int discretes[] = {1,0};
    narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
    
    if (m_bModelSelection)
    {
        cout << "Narrow model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));

            // Narrow search
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "boost_" << m_distsToMargin.size() << "_" << m_boostType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat goodnesses;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "boost_" << m_distsToMargin.size() << "_" << m_boostType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (goodnesses.empty()) goodnesses = aux.col(params.size());
        else cv::hconcat(goodnesses, aux.col(params.size()), goodnesses);
    }
//    cv::reduce(goodnesses, aux, 1, CV_REDUCE_AVG);
//    cv::hconcat(coarseExpandedParameters, aux, goodnesses);
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    fusionPredictions.create(m_responses.rows, 1, cv::DataType<int>::type);
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodness;
        if (m_bGlobalBest)
        {
            GridMat globalMean;
            cv::reduce(goodnesses, goodness, 1, CV_REDUCE_AVG);
        }
        else
        {
            goodness = goodnesses.col(k);
        }
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodness, &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestNumOfWeaks     = narrowExpandedParameters.row(best.y).at<float>(0,0);
        float bestWeightTrimRate = narrowExpandedParameters.row(best.y).at<float>(0,1);
        
        CvBoostParams boostParams (m_boostType, bestNumOfWeaks, bestWeightTrimRate, 1, 0, NULL);
        
        m_pClassifier->train(validTrData, CV_ROW_SAMPLE, validTrResponses,
                             cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
                             boostParams);
        
        // Test phase
        cv::Mat tePredictions (teResponses.rows, teResponses.cols, teResponses.type());
        for (int d = 0; d < teData.rows; d++)
            tePredictions.at<int>(d,0) = (int) m_pClassifier->predict(teData.row(d));
        
        cvx::setMat(tePredictions, fusionPredictions, m_partitions == k);
    }
    cout << endl;
    
    // Mat to GridMat
    m_fusionPredictions = fusionPredictions;
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::_modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies)
{
    cv::Mat foldAccs (expandedParams.rows, 1, cv::DataType<float>::type); // results
    
    for (int m = 0; m < expandedParams.rows; m++)
    {
        // Training phase
        cv::Mat selectedParams = expandedParams.row(m);
        
        float bestNumOfWeaks = selectedParams.at<float>(0,0);
        float bestWeightTrimRate = selectedParams.at<float>(0,1); // indeed, gamma not used if not RBF kernel
        
        CvBoostParams boostParams (m_boostType, (int) bestNumOfWeaks, bestWeightTrimRate, 1, 0, NULL);
        CvBoost classifier (descriptorsTr, CV_ROW_SAMPLE, responsesTr,
                             cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
                             boostParams);
        
        // Test phase
        cv::Mat predictionsVal (responsesVal.rows, responsesVal.cols, responsesVal.type());
        for (int d = 0; d < descriptorsVal.rows; d++)
        {
            int prediction = classifier.predict(descriptorsVal.row(d));
            predictionsVal.at<int>(d,0) = prediction;
        }
        
        // Compute an accuracy measure
        foldAccs.at<float>(m,0) = accuracy(responsesVal, predictionsVal);
    }
    
    m_mutex.lock();
    foldAccs.copyTo(accuracies.col(k));
    m_mutex.unlock();
}

void ClassifierFusionPrediction<cv::EM40,CvBoost>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, m_modelSelecK, cv::DataType<float>::type);
    
    cout << "(";
    boost::thread_group tg;
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k;
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        tg.add_thread(new boost::thread( boost::bind(&ClassifierFusionPrediction::_modelSelection, this, trSbjObjData, trSbjObjResponses, valData, valResponses, k, expandedParams, accuracies) ));
    }
    tg.join_all();
    cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}

// CvANN_MLP

ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::ClassifierFusionPrediction()
: m_numOfEpochs(500), m_numOfRepetitions(5)
{
    
}

cv::Mat ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::encode(cv::Mat classes)
{
    cv::Mat mat (classes.rows, 2, cv::DataType<float>::type, cv::Scalar(0));
    
    for (int i = 0; i < classes.rows; i++)
    {
        int idx = classes.at<int>(i,0);
        mat.at<float>(i, idx) = 1.0f;
    }
    
    return mat;
}

cv::Mat ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::decode(cv::Mat predictions)
{
    cv::Mat mat (predictions.rows, 1, cv::DataType<int>::type);
    
    for (int i = 0; i < predictions.rows; i++)
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(predictions.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        mat.at<int>(i,0) = maxLoc.x;
    }
    
    return mat;
}

void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::setActivationFunctionType(int type)
{
    m_actFcnType = type;
}

void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::setHiddenLayerSizes(vector<float> hiddenSizes)
{
    m_hiddenLayerSizes = hiddenSizes;
}

//void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::setBackpropDecayWeightScales(vector<float> dwScales)
//{
//    m_bpDwScales = dwScales;
//}
//
//void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::setBackpropMomentScales(vector<float> momScales)
//{
//    m_bpMomentScales = momScales;
//}

void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::predict(cv::Mat& fusionPredictions)
{
    m_pClassifier->clear();
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_hiddenLayerSizes);
    
    // create a list of parameters' variations
    cv::Mat coarseExpandedParameters;
    expandParameters(params, coarseExpandedParameters);
    
    if (m_bModelSelection)
    {
        cout << "Coarse model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
            
            // Coarse search
            cv::Mat coarseGoodnesses; // for instance: accuracies
            modelSelection(trData, trResponses, coarseExpandedParameters, coarseGoodnesses);
            
            std::stringstream coarsess;
            coarsess << "mlp_" << m_distsToMargin.size() << "_" << m_actFcnType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(coarseExpandedParameters, coarseGoodnesses, coarseGoodnesses);
            cvx::save(coarsess.str(), coarseGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat coarseGoodnesses, aux;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "mlp_" << m_distsToMargin.size() << "_" << m_actFcnType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (coarseGoodnesses.empty()) coarseGoodnesses = aux.col(params.size());
        else cv::hconcat(coarseGoodnesses, aux.col(params.size()), coarseGoodnesses);
    }
    cv::reduce(coarseGoodnesses, aux, 1, CV_REDUCE_AVG);
    cv::hconcat(coarseExpandedParameters, aux, coarseGoodnesses);
    
    cv::Mat narrowExpandedParameters;
    int discretes[] = {1};
    narrow<float>(coarseExpandedParameters, coarseGoodnesses, m_narrowSearchSteps, discretes, narrowExpandedParameters);
    
    if (m_bModelSelection)
    {
        cout << "Narrow model selection CVs [" << m_testK << "]: " << endl;
        
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
            cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
            cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
            cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
            cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
            
            // Narrow search
            cv::Mat narrowGoodnesses;
            modelSelection(trData, trResponses, narrowExpandedParameters, narrowGoodnesses);
            
            std::stringstream narrowss;
            narrowss << "mlp_" << m_distsToMargin.size() << "_" << m_actFcnType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
            cv::hconcat(narrowExpandedParameters, narrowGoodnesses, narrowGoodnesses);
            cvx::save(narrowss.str(), narrowGoodnesses);
        }
        cout << endl;
    }
    
    cv::Mat goodnesses;
    for (int k = 0; k < m_testK; k++)
    {
        cv::Mat aux;
        std::stringstream ss;
        ss << "mlp_" << m_distsToMargin.size() << "_" << m_actFcnType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << (m_bTrainMirrored ? "m" : "") << ".yml";
        cvx::load(ss.str(), aux);
        if (goodnesses.empty()) goodnesses = aux.col(params.size());
        else cv::hconcat(goodnesses, aux.col(params.size()), goodnesses);
    }
//    cv::reduce(goodnesses, aux, 1, CV_REDUCE_AVG);
//    cv::hconcat(coarseExpandedParameters, aux, goodnesses);
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    fusionPredictions.create(m_responses.rows, 1, cv::DataType<int>::type);
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teData = cvx::indexMat(m_data, m_partitions == k);
        cv::Mat valData = cvx::indexMat(m_data, m_partitions == ((k+1) % m_testK));
        cv::Mat trResponses = cvx::indexMat(m_responses, (m_partitions != k) & (m_partitions != ((k+1) % m_testK)));
        cv::Mat teResponses = cvx::indexMat(m_responses, m_partitions == k);
        cv::Mat valResponses = cvx::indexMat(m_responses, m_partitions == ((k+1) % m_testK));
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodness;
        if (m_bGlobalBest)
        {
            GridMat globalMean;
            cv::reduce(goodnesses, goodness, 1, CV_REDUCE_AVG);
        }
        else
        {
            goodness = goodnesses.col(k);
        }
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point worst, best;
        cv::minMaxLoc(goodness, &minVal, &maxVal, &worst, &best);
        
        // Training phase
        float bestHiddenSize    = narrowExpandedParameters.row(best.y).at<float>(0,0);
        
        cv::Mat layerSizes (3, 1, cv::DataType<int>::type);
        layerSizes.at<int>(0,0) = trData.cols; // as many inputs as feature vectors' num of dimensions
        layerSizes.at<int>(1,0) = bestHiddenSize; // num of hidden neurons experimentally selected
        layerSizes.at<int>(2,0) = 2; // one output neuron
        
        CvANN_MLP classifier;
        classifier.create(layerSizes, m_actFcnType);
        
        CvANN_MLP_TrainParams mlpParams (cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1, 1e-2), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
                
        classifier.train(validTrData, encode(validTrResponses), cv::Mat(), cv::Mat(),
                             mlpParams);
        
        cv::Mat valPredictions;
        classifier.predict(valData, valPredictions);

        float bestValAcc, prevAcc;
        prevAcc = bestValAcc = accuracy(valResponses, decode(valPredictions));
        
        cv::Mat tePredictions;
        classifier.predict(teData, tePredictions);
        
        bool overfitting = false; // wheter performance in validation keeps decreasing
        int counter = 0; // number of iterations the performance in validation is getting worse
        bool saturation = false; // performance in training is saturated
        
        int e;
        for (e = 0; e < m_numOfEpochs && !overfitting && !saturation; e++)
        {
            int iters = classifier.train(validTrData, encode(validTrResponses), cv::Mat(), cv::Mat(),
                                             mlpParams, CvANN_MLP::UPDATE_WEIGHTS);
            saturation = (iters < 1);
            
            // Test phase
            cv::Mat valPredictions;
            classifier.predict(valData, valPredictions);
            
            float acc = accuracy(valResponses, decode(valPredictions));
            if (acc > bestValAcc)
            {
                bestValAcc = acc;
                classifier.predict(teData, tePredictions);
            }
            
            if (acc > prevAcc) counter = 0;
            else overfitting = (++counter == 3); // last e consecutive epochs getting worse
            
            prevAcc = acc;
        }

        cvx::setMat(decode(tePredictions), fusionPredictions, m_partitions == k);
    }
    cout << endl;
    
    m_fusionPredictions = fusionPredictions;
}

void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::_modelSelection(cv::Mat& descriptorsTr, cv::Mat& responsesTr, cv::Mat& descriptorsVal, cv::Mat& responsesVal, int k, cv::Mat& expandedParams, cv::Mat& accuracies)
{
    cv::Mat foldAccs (expandedParams.rows, m_numOfRepetitions, cv::DataType<float>::type); // results
    
    for (int m = 0; m < expandedParams.rows; m++)
    {
        // Training phase
        cv::Mat selectedParams = expandedParams.row(m);
        
        float hiddenSize     = selectedParams.at<float>(0,0);
        
        cv::Mat layerSizes (3, 1, cv::DataType<int>::type);
        layerSizes.at<int>(0,0) = descriptorsTr.cols; // as many inputs as feature vectors' num of dimensions
        layerSizes.at<int>(1,0) = hiddenSize; // num of hidden neurons experimentally selected
        layerSizes.at<int>(2,0) = 2; // one output neuron
        
        for (int r = 0; r < m_numOfRepetitions; r++)
        {
            CvANN_MLP classifer (layerSizes, m_actFcnType);
            
            CvANN_MLP_TrainParams mlpParams (cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1, 1e-2), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
            
            classifer.train(descriptorsTr, encode(responsesTr), cv::Mat(), cv::Mat(),
                                 mlpParams);
            
            cv::Mat predictionsVal;
            classifer.predict(descriptorsVal, predictionsVal);
            
            float prevAcc, bestAcc;
            prevAcc = bestAcc = accuracy(responsesVal, decode(predictionsVal));
            
            bool overfitting = false; // wheter performance in validation keeps decreasing
            int counter = 0; // number of iterations the performance in validation is getting worse
            
            bool saturation = false; // performance in training is saturated
            
            int e;
            for (e = 0; e < m_numOfEpochs && !overfitting && !saturation; e++)
            {
                int iters = classifer.train(descriptorsTr, encode(responsesTr), cv::Mat(), cv::Mat(),
                                                 mlpParams, CvANN_MLP::UPDATE_WEIGHTS);
                saturation = (iters < 1);
                
                // Test phase
                cv::Mat predictionsVal;
                classifer.predict(descriptorsVal, predictionsVal);
                
                float acc = accuracy(responsesVal, decode(predictionsVal));
                if (acc > bestAcc) bestAcc = acc;
                
                if (acc > prevAcc) counter = 0;
                else overfitting = (++counter == 3);
                
                prevAcc = acc;
            }
            
            // Compute an accuracy measure
            foldAccs.at<float>(m,r) = bestAcc;
        }
    }
    
    cv::Mat tmp;
    cvx::hmean(foldAccs, tmp);
    
    m_mutex.lock();
    tmp.copyTo(accuracies.col(k));
    m_mutex.unlock();
}

void ClassifierFusionPrediction<cv::EM40,CvANN_MLP>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, m_modelSelecK, cv::DataType<float>::type);
    
    cout << "(";
    boost::thread_group tg;
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k;
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        tg.add_thread(new boost::thread( boost::bind(&ClassifierFusionPrediction::_modelSelection, this, trSbjObjData, trSbjObjResponses, valData, valResponses, k, expandedParams, accuracies) ));
    }
    tg.join_all();
    cout << ") " << endl;
    
    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}