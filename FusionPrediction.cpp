//
//  FusionPrediction.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/04/14.
//
//

#include "FusionPrediction.h"
#include "StatTools.h"

// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setData(vector<GridMat> distsToMargin, vector<GridMat> predictions);

template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setResponses(cv::Mat);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModelSelection(bool flag);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setModelSelectionParameters(int, bool);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setValidationParameters(int, int);
template void ClassifierFusionPredictionBase<cv::EM,CvSVM>::setStackedPrediction(bool flag);
// -----------------------------------------------------------------------------


SimpleFusionPrediction<cv::EM>::SimpleFusionPrediction()
{
    
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, GridMat& fusedPredictions)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat predictions (allPredictions[0]);
    GridMat distsToMargin (allDistsToMargin[0]);
    
    for (int i = 1; i < allPredictions.size(); i++)
    {
        predictions.hconcat(allPredictions[i]);
        distsToMargin.hconcat(allDistsToMargin[i]);
    }
    
    int hp = predictions.crows();
    int wp = predictions.ccols();
    int n = predictions.at(0,0).rows;
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        assert (n == predictions.at(i,j).rows);
    }
    
    fusedPredictions.create<int>(hp, wp, n, 1);
    
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            int pos = 0;
            int neg = 0;
            float accPosDists = 0;
            float accNegDists = 0;
            for (int m = 0; m < predictions.at(i,j).cols; m++)
            {
                float dist = distsToMargin.at<float>(i,j,k,m);
                if (predictions.at<int>(i,j,k,m) == 0)
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
                fusedPredictions.at<int>(i,j,k,0) = 0;
            else if (pos > neg)
                fusedPredictions.at<int>(i,j,k,0) = 1;
            else
                fusedPredictions.at<int>(i,j,k,0) = accPosDists/pos > abs(accNegDists/pos);

        }
        
    }
}


void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allPredictions, vector<GridMat> allDistsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat predictions (allPredictions[0]);
    GridMat distsToMargin (allDistsToMargin[0]);
    
    for (int i = 1; i < allPredictions.size(); i++)
    {
        predictions.hconcat(allPredictions[i]);
        distsToMargin.hconcat(allDistsToMargin[i]);
    }
    
    int hp = predictions.crows();
    int wp = predictions.ccols();
    int n = predictions.at(0,0).rows;
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        assert (n == predictions.at(i,j).rows);
    }
    
    fusedPredictions.create<int>(hp, wp, n, 1);
    fusedDistsToMargin.create<float>(hp, wp, n, 1);
    
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            int pos = 0;
            int neg = 0;
            float accPosDists = 0;
            float accNegDists = 0;
            for (int m = 0; m < predictions.at(i,j).cols; m++)
            {
                float dist = distsToMargin.at<float>(i,j,k,m);
                if (predictions.at<int>(i,j,k,m) == 0)
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
                fusedPredictions.at<int>(i,j,k,0) = 0;
                fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
            }
            else if (pos > neg)
            {
                fusedPredictions.at<int>(i,j,k,0) = 1;
                fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
            }
            else
            {
                float meanNeg = accNegDists/neg;
                float meanPos = accPosDists/pos;
                if (meanPos > abs(meanNeg))
                {
                    fusedPredictions.at<int>(i,j,k,0) = 1;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
                }
                else
                {
                    fusedPredictions.at<int>(i,j,k,0) = 0;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
                }
            }
        }
        
    }
    
//    GridMat predictions (allPredictions[0]);
//    GridMat distsToMargin (allDistsToMargin[0]);
//    
//    for (int i = 1; i < allPredictions.size(); i++)
//    {
//        predictions.hconcat(allPredictions[i]);
//        distsToMargin.hconcat(allDistsToMargin[i]);
//    }
//    
//    int elements = predictions.at(0,0).rows;
//    for (int i = 0; i < predictions.crows(); i++) for (int j = 1; j < predictions.ccols(); j++)
//    {
//        assert (elements == predictions.at(i,j).rows);
//    }
//    
//    GridMat negMask (predictions == 0);
//    GridMat posMask (predictions == 1);
//    negMask = negMask / 255;
//    posMask = posMask / 255;
//    
//    GridMat hAccNegMask, hAccPosMask;
//    negMask.sum(hAccNegMask, 1);
//    posMask.sum(hAccPosMask, 1);
//    
//    GridMat ones (predictions.crows(), predictions.ccols(), elements, 1, 1);
//
//    int majorityVal = ceil(allPredictions.size()/2);
//    ones.copyTo(fusedPredictions, hAccPosMask > majorityVal);
//    
//    GridMat drawsMask = (hAccNegMask <= majorityVal) & (hAccPosMask <= majorityVal);
//    GridMat drawDistsToMargin (distsToMargin, drawsMask);
//    
//    GridMat drawFusedPredictions;
//    compute(drawDistsToMargin, drawFusedPredictions);
//    
//    fusedPredictions.set(drawFusedPredictions, drawsMask);
}

void SimpleFusionPrediction<cv::EM>::compute(vector<GridMat> allDistsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    // Concatenate along the horizontal direction all the modalities' predictions in a GridMat,
    // and the same for the distsToMargin
    
    GridMat hConcatDistsToMargin;
    
    for (int i = 0; i < allDistsToMargin.size(); i++)
    {
        hConcatDistsToMargin.hconcat(allDistsToMargin[i]);
    }
    
    compute(hConcatDistsToMargin, fusedPredictions, fusedDistsToMargin);
}

void SimpleFusionPrediction<cv::EM>::compute(GridMat distsToMargin, GridMat& fusedPredictions, GridMat& fusedDistsToMargin)
{
    int hp = distsToMargin.crows();
    int wp = distsToMargin.ccols();
    int n = distsToMargin.at(0,0).rows;
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        assert (n == distsToMargin.at(i,j).rows);
    }
    
    fusedPredictions.create<int>(hp, wp, n, 1);
    fusedDistsToMargin.create<float>(hp, wp, n, 1);
    
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
        {
            int pos = 0;
            int neg = 0;
            float accPosDists = 0;
            float accNegDists = 0;
            for (int m = 0; m < distsToMargin.at(i,j).cols; m++)
            {
                float dist = distsToMargin.at<float>(i,j,k,m);
                if (distsToMargin.at<int>(i,j,k,m) < 0)
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
                fusedPredictions.at<int>(i,j,k,0) = 0;
                fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
            }
            else if (pos > neg)
            {
                fusedPredictions.at<int>(i,j,k,0) = 1;
                fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
            }
            else
            {
                float meanNeg = accNegDists/neg;
                float meanPos = accPosDists/pos;
                if (meanPos > abs(meanNeg))
                {
                    fusedPredictions.at<int>(i,j,k,0) = 1;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accPosDists/pos;
                }
                else
                {
                    fusedPredictions.at<int>(i,j,k,0) = 0;
                    fusedDistsToMargin.at<float>(i,j,k,0) = accNegDists/neg;
                }
            }
        }
        
    }

//    GridMat negMask (distsToMargin < 0);
//    GridMat posMask (distsToMargin > 0);
//    
//    GridMat negDistsToMargin, posDistsToMargin;
//    distsToMargin.copyTo(negDistsToMargin, negMask);
//    distsToMargin.copyTo(posDistsToMargin, posMask);
//    
//    GridMat hAccNegMsk, hAccPosMsk, hAccNegDistsToMargin, hAccPosDistsToMargin;
//    negMask.sum(hAccNegMsk, 1);
//    posMask.sum(hAccPosMsk, 1);
//    negDistsToMargin.sum(hAccNegDistsToMargin, 1); // 1 indicates in the horizontal direction
//    posDistsToMargin.sum(hAccPosDistsToMargin, 1);
//    
//    GridMat hMeanNegDistsToMargin, hMeanPosDistsToMargin;
//    hMeanNegDistsToMargin = hAccNegDistsToMargin / hAccNegMsk;
//    hMeanPosDistsToMargin = hAccPosDistsToMargin / hAccPosMsk;
//    
//    fusedPredictions = (hMeanPosDistsToMargin > hMeanNegDistsToMargin.abs()) / 255;
}

//
// ClassifierFusionPredictionBase class
//

template<typename ClassifierT>
ClassifierFusionPredictionBase<cv::EM, ClassifierT>::ClassifierFusionPredictionBase()
: m_pClassifier(new ClassifierT), m_bModelSelection(true), m_bStackPredictions(false), m_narrowSearchSteps(7)
{
    
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<GridMat> distsToMargin)
{
    m_distsToMargin = distsToMargin;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setData(vector<GridMat> distsToMargin, vector<GridMat> predictions)
{
    m_distsToMargin = distsToMargin;
    m_predictions = predictions;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setResponses(cv::Mat responses)
{
    m_responses = responses;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelSelection(bool flag)
{
    m_bModelSelection = flag;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setModelSelectionParameters(int k, bool best)
{
    m_modelSelecK = k;
    m_selectBest = best;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::setValidationParameters(int k, int seed)
{
    m_testK = k;
    m_seed = seed;
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM, ClassifierT>::formatData()
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
            cv::Mat aux;
            m_predictions[i].at(0,0).convertTo(aux, m_data.type());
            cv::hconcat(m_data, aux, m_data);
        }
    }
}

template<typename ClassifierT>
void ClassifierFusionPredictionBase<cv::EM,ClassifierT>::setStackedPrediction(bool flag)
{
    m_bStackPredictions = flag;
}


//
// ClassifierFusionPrediction class templates' specialization
//

ClassifierFusionPrediction<cv::EM,CvSVM>::ClassifierFusionPrediction()
: m_numItersSVM(5000)
{
    
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setKernelType(int type)
{
    m_kernelType = type;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setCs(vector<float> cs)
{
    m_cs = cs;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::setGammas(vector<float> gammas)
{
    m_gammas = gammas;
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::compute(GridMat& gpredictions)
{
    formatData();
    
    // Prepare parameters' combinations
    vector<vector<float> > params;
    params.push_back(m_cs);
    if (m_kernelType == CvSVM::RBF)
        params.push_back(m_gammas);
    
    // create a list of parameters' variations
    cv::Mat expandedParameters;
    expandParameters(params, expandedParameters);
    
    cv::Mat partitions;
    cvpartition(m_responses, m_testK, m_seed, partitions);
    
    if (m_bModelSelection)
    {
        cout << "Model selection CVs [" << m_testK << "]: " << endl;
        
        cv::Mat goodnesses; // for instance: accuracies
        for (int k = 0; k < m_testK; k++)
        {
            cout << k << " " << endl;
            
            cv::Mat trData = cvx::indexMat(m_data, partitions != k);
            cv::Mat teData = cvx::indexMat(m_data, partitions == k);
            cv::Mat trResponses = cvx::indexMat(m_responses, partitions != k);
            cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
            
            // Coarse search
            modelSelection(trData, trResponses, expandedParameters, goodnesses);
            
            std::stringstream coarsess;
            coarsess << "svm_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_coarse-goodnesses_" << k << ".yml";
            cv::hconcat(expandedParameters, goodnesses, goodnesses);
            cvx::save(coarsess.str(), goodnesses);
            
            // Find best parameters (using goodnesses) to train the final model
            double minVal, maxVal;
            cv::Point min, max;
            cv::minMaxLoc(goodnesses.col(params.size()), &minVal, &maxVal, &min, &max);
            
            int idxC = max.y;
            
            int idxInfC = idxC;
            int idxSupC = idxC;
            if (idxC > 0) idxInfC = idxC - 1;
            if (idxC < params[0].size() - 1) idxSupC = idxC + 1;
            
            int idxGamma, idxInfGamma, idxSupGamma;
            if (m_kernelType == CvSVM::RBF)
            {
                idxC = max.y / params[1].size();
            
                if (idxC > 0) idxInfC = idxC - 1;
                if (idxC < params[0].size() - 1) idxSupC = idxC + 1;
            
                idxGamma = max.y % params[0].size();
            
                idxInfGamma = idxGamma;
                idxSupGamma = idxGamma;
                if (idxGamma > 0) idxInfGamma = idxGamma - 1;
                if (idxGamma < params[1].size() - 1) idxSupGamma = idxGamma + 1;
            }
            
            vector<vector<float> > narrowParams;
            vector<float> narrowCs, narrowGammas;
            
            int steps = (m_narrowSearchSteps % 2 == 0) ? m_narrowSearchSteps + 1 : m_narrowSearchSteps;
            cvx::linspace(params[0][idxInfC], params[0][idxSupC], (idxInfC != idxC) && (idxSupC != idxC) ? steps : steps/2 + 1, narrowCs);
            narrowParams.push_back(narrowCs);
            
            if (m_kernelType == CvSVM::RBF)
            {
                cvx::linspace(params[1][idxInfGamma], params[1][idxSupGamma], (idxGamma != idxInfGamma) && (idxGamma != idxSupGamma) ? steps : steps/2 + 1, narrowGammas);
                narrowParams.push_back(narrowGammas);
            }
            
            cv::Mat narrowExpandedParameters;
            expandParameters(narrowParams, narrowExpandedParameters);

            // Narrow search
            modelSelection(trData, trResponses, narrowExpandedParameters, goodnesses);
            
            std::stringstream narrowss;
            narrowss << "svm_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
            cv::hconcat(narrowExpandedParameters, goodnesses, goodnesses);
            cvx::save(narrowss.str(), goodnesses);
        }
        cout << endl;
    }
    
    
    cout << "Out-of-sample CV [" << m_testK << "] : " << endl;
    
    cv::Mat predictions;
    
    for (int k = 0; k < m_testK; k++)
    {
        cout << k << " ";
        
        cv::Mat trData = cvx::indexMat(m_data, partitions != k);
        cv::Mat teData = cvx::indexMat(m_data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(m_responses, partitions != k);
        cv::Mat teResponses = cvx::indexMat(m_responses, partitions == k);
        
        cv::Mat validTrData = cvx::indexMat(trData, trResponses >= 0); // -1 labels not used in training
        cv::Mat validTrResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat goodnesses;
        std::stringstream ss;
        ss << "svm_" << m_kernelType << (m_bStackPredictions ? "_s" : "") << "_narrow-goodnesses_" << k << ".yml";
        cvx::load(ss.str(), goodnesses);
        
        // Find best parameters (using goodnesses) to train the final model
        double minVal, maxVal;
        cv::Point min, max;
        cv::minMaxLoc(goodnesses.col(params.size()), &minVal, &maxVal, &min, &max);
        
        // Training phase
        cv::Mat bestParams = goodnesses.row(max.y);
        float bestC = bestParams.at<float>(0,0);
        float bestGamma = bestParams.at<float>(0,1);
        
        CvSVMParams svmParams (CvSVM::C_SVC, m_kernelType, 0, bestGamma, 0, bestC, 0, 0, 0,
                            cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, FLT_EPSILON ));
        m_pClassifier->train(validTrData, validTrResponses, cv::Mat(), cv::Mat(), svmParams);
        
        // Prediction phase
        cv::Mat tePredictions;
        m_pClassifier->predict(teData, tePredictions);
        
        cvx::setMat(tePredictions, predictions, partitions == k);
    }
    
    // Mat to GridMat
    gpredictions.setTo(predictions);
}

void ClassifierFusionPrediction<cv::EM,CvSVM>::modelSelection(cv::Mat data, cv::Mat responses, cv::Mat expandedParams, cv::Mat& goodnesses)
{
    goodnesses.release();
    
    // Partitionate the data in folds
    cv::Mat partitions;
    cvpartition(responses, m_modelSelecK, m_seed, partitions);
    
    cv::Mat accuracies (expandedParams.rows, 0, cv::DataType<float>::type);
    
    cout << "(";
    for (int k = 0; k < m_modelSelecK; k++)
    {
        cout << k << " ";
        
        // Get fold's data
        
        cv::Mat trData = cvx::indexMat(data, partitions != k);
        cv::Mat valData = cvx::indexMat(data, partitions == k);
        cv::Mat trResponses = cvx::indexMat(responses, partitions != k);
        cv::Mat valResponses = cvx::indexMat(responses, partitions == k);
        
        cv::Mat trSbjObjData = cvx::indexMat(trData, trResponses >= 0); // ignore unknown category (class -1) in training
        cv::Mat trSbjObjResponses = cvx::indexMat(trResponses, trResponses >= 0);
        
        cv::Mat foldAccs (expandedParams.rows, 1, cv::DataType<float>::type); // results
        
        for (int m = 0; m < expandedParams.rows; m++)
        {
            // Training phase
            cv::Mat selectedParams = expandedParams.row(m);
            
            float C = selectedParams.at<float>(0,0);
            
            float gamma = 0;
            if (m_kernelType == CvSVM::RBF)
                gamma = selectedParams.at<float>(0,1); // indeed, gamma not used if not RBF kernel
                
            CvSVMParams params (CvSVM::C_SVC, m_kernelType, 0, gamma, 0, C, 0, 0, 0,
                                cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, m_numItersSVM, FLT_EPSILON ));
            
            m_pClassifier->train(trSbjObjData, trSbjObjResponses, cv::Mat(), cv::Mat(), params);
            
            // Test phase
            cv::Mat valPredictions;
            m_pClassifier->predict(valData, valPredictions);
        
            // Compute an accuracy measure
            foldAccs.at<float>(m,0) = accuracy(valResponses, valPredictions);
        }
        
        if (accuracies.cols == 0)
            accuracies = foldAccs;
        else
            cv::hconcat(accuracies, foldAccs, accuracies); // concatenate along the horizontal direction
    }
    cout << ") " << endl;

    // mean along the horizontal direction
    cvx::hmean(accuracies, goodnesses); // one column of m accuracies evaluation the m combinations is left
}