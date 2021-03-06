//
//  GridPredictor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#include "GridPredictor.h"
#include "StatTools.h"
#include "CvExtraTools.h"


template GridPredictorBase<cv::EM40>::~GridPredictorBase();
template void GridPredictorBase<cv::EM40>::setDimensionalityReduction(cv::Mat variances);

//
// GridPredictorBase
//

template<typename PredictorT>
GridPredictorBase<PredictorT>::GridPredictorBase(int hp, int wp)
: m_hp(hp), m_wp(wp), m_bDimReduction(false)
{
    m_pPredictors.resize(m_hp * m_wp);
    m_pPCAs.resize(m_hp * m_wp);
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        m_pPredictors[i * m_wp + j] = new PredictorT();
        m_pPCAs[i * m_wp + j] = new cv::PCA();
    }
}

template<typename PredictorT>
PredictorT* GridPredictorBase<PredictorT>::at(unsigned int i, unsigned int j)
{
    return m_pPredictors[i * m_wp + j];
}

template<typename PredictorT>
cv::PCA* GridPredictorBase<PredictorT>::getPCA(unsigned int i, unsigned int j)
{
    return m_pPCAs[i * m_wp + j];
}

template<typename PredictorT>
void GridPredictorBase<PredictorT>::setDimensionalityReduction(cv::Mat variances)
{
    m_bDimReduction = true;
    m_variances = variances;
}

template<typename PredictorT>
GridPredictorBase<PredictorT>::~GridPredictorBase()
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        delete m_pPredictors[i * m_wp + j];
        delete m_pPCAs[i * m_wp + j];
    }
}

//
// GridPredictor<PredictorT>
//


GridPredictor<cv::EM40>::GridPredictor(int hp, int wp)
: GridPredictorBase<cv::EM40>(hp, wp)
{
}

//void GridPredictor<cv::EM40>::setParameters(GridMat parameters)
//{
//    m_nmixtures.release();
//    m_logthreshold.release();
//    m_nmixtures.create(m_hp, m_wp, cv::DataType<int>::type);
//    m_logthreshold.create(m_hp, m_wp, cv::DataType<int>::type);
//    
//    if (!m_data.isEmpty())
//    {
//        m_projData.release();
//        m_projData.create(m_hp, m_wp);
//    }
//    
//    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
//    {
//        m_nmixtures.at<int>(i,j) = parameters.at<int>(i,j,0,0);
//        at(i,j)->set("nclusters", parameters.at<int>(i,j,0,0));
//        
//        m_logthreshold.at<int>(i,j) = parameters.at<int>(i,j,0,1);
//        
//        if (!m_data.isEmpty())
//            cvx::computePCA(m_data.at(i,j), *getPCA(i,j),
//                            m_projData.at(i,j), CV_PCA_DATA_AS_ROW, m_variances.at<double>(i,j));
//    }
//}

void GridPredictor<cv::EM40>::setNumOfMixtures(cv::Mat nmixtures)
{
    m_nmixtures = nmixtures;
    
    for (int i = 0; i < nmixtures.rows; i++) for (int j = 0; j < nmixtures.cols; j++)
    {
        at(i,j)->set("nclusters", nmixtures.at<float>(i,j));
    }
}

void GridPredictor<cv::EM40>::setEpsilons(cv::Mat epsilons)
{
    m_epsilons = epsilons;
    
    for (int i = 0; i < epsilons.rows; i++) for (int j = 0; j < epsilons.cols; j++)
    {
        at(i,j)->set("epsilon", epsilons.at<float>(i,j));
    }
}

void GridPredictor<cv::EM40>::setLoglikelihoodThreshold(cv::Mat loglikes)
{
    m_logthreshold = loglikes;
}

void GridPredictor<cv::EM40>::train(GridMat data)
{
    m_data = data;
    
    m_projData.create(m_hp, m_wp);
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat cellData;
        if (!m_bDimReduction)
        {
            cellData = m_data.at(i,j);
        }
        else
        {
            cvx::computePCA(m_data.at(i,j), *getPCA(i,j),
                            cellData, CV_PCA_DATA_AS_ROW, m_variances.at<double>(i,j));
            m_projData.at(i,j) = cellData;
        }
        
        at(i,j)->train(cellData);
    }
}

/*
 * Returns predictions of the cells, the normalized loglikelihoods [0,1]
 */
void GridPredictor<cv::EM40>::predict(GridMat data, GridMat& loglikelihoods)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat& cell = data.at(i,j);
        cv::Mat cellLoglikelihoods (cell.rows, 1, cv::DataType<float>::type);
        
        for (int d = 0; d < cell.rows; d++)
        {
            cv::Mat descriptor = cell.row(d);
            
            if (m_bDimReduction)
                descriptor = getPCA(i,j)->project(descriptor);
            
            cv::Vec3d res = at(i,j)->predict(descriptor);
            
            cellLoglikelihoods.at<float>(d,0) = static_cast<float>(res.val[1]);
        }
        
        cv::Mat stdCellLoglikelihoods;
        cv::Scalar mean, stddev;
        cv::meanStdDev(cellLoglikelihoods, mean, stddev);
        stdCellLoglikelihoods = (cellLoglikelihoods - mean.val[0]) / stddev.val[0];
    
        loglikelihoods.assign(stdCellLoglikelihoods, i, j);
    }
}


/*
 * Returns predictions of the cells, the normalized loglikelihoods [0,1]
 */
void GridPredictor<cv::EM40>::predict(GridMat data, GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat& cell = data.at(i,j);
        cv::Mat_<int> cellLabels(cell.rows, 1);
        cv::Mat_<float> cellLoglikelihoods (cell.rows, 1);
        
        for (int d = 0; d < cell.rows; d++)
        {
            cv::Mat descriptor = cell.row(d);
            
            if (m_bDimReduction)
                descriptor = getPCA(i,j)->project(descriptor);
            
            cv::Vec3d res = at(i,j)->predict(descriptor);

            cellLoglikelihoods.at<float>(d,0) = static_cast<float>(res.val[1]); // res.val[0] the global likelihood, res.val[0] the likelihood in the cluster
            cellLabels.at<int>(d,0) = static_cast<int>(res.val[2]);
        }

        // Standardized loglikelihoods
        cv::Mat_<float> means, stddevs;
        means.create(cellLoglikelihoods.rows, cellLoglikelihoods.cols);
        stddevs.create(cellLoglikelihoods.rows, cellLoglikelihoods.cols);
        for (int l = 0; l < at(i,j)->get<int>("nclusters"); l++)
        {
            cv::Scalar mean, stddev;
            cv::meanStdDev(cellLoglikelihoods, mean, stddev, cellLabels == l);
            means.setTo(mean.val[0], cellLabels == l);
            stddevs.setTo(stddev.val[0], cellLabels == l);
        }
        cv::Mat_<float> ctrCellLoglikelihoods, stdCellLoglikelihoods;
        cv::subtract(cellLoglikelihoods, means, ctrCellLoglikelihoods);
        cv::divide(ctrCellLoglikelihoods, stddevs, stdCellLoglikelihoods);
        
        // Predictions evaluation comparing the standardized loglikelihoods to a threshold,
        // loglikelihoods over threshold are considered subject (1)
        cv::Mat cellPredictions;
        cv::threshold(stdCellLoglikelihoods, cellPredictions, m_logthreshold.at<float>(i,j), 1, CV_THRESH_BINARY);
        cellPredictions.convertTo(cellPredictions, cv::DataType<int>::type);
        
        // Center the values around the loglikelihood threshold, so as to have
        // subjects' margin > 0 and objects' margin < 0. And scale to take into
        // accound the variance of the dists' sample
        cv::Mat_<float> diffs = stdCellLoglikelihoods - m_logthreshold.at<float>(i,j); // center
        cv::Mat_<float> powers;
        cv::pow(diffs, 2, powers);
        float scale = sqrt(cv::sum(powers).val[0] / stdCellLoglikelihoods.rows);
        cv::Mat_<float> cellsDistsToMargin = diffs / scale; // scale
        
        predictions.assign(cellPredictions, i, j);
        loglikelihoods.assign(stdCellLoglikelihoods, i, j);
        distsToMargin.assign(cellsDistsToMargin, i, j);
    }
}
