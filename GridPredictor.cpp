//
//  GridPredictor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#include "GridPredictor.h"
#include "StatTools.h"


//
// GridPredictorBase
//

template<typename PredictorT>
GridPredictorBase<PredictorT>::GridPredictorBase(int hp, int wp)
: m_hp(hp), m_wp(wp)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        m_predictors[i * m_wp + j] = new PredictorT();
    }
}

template<typename PredictorT>
PredictorT* GridPredictorBase<PredictorT>::at(unsigned int i, unsigned int j)
{
    if (m_predictors.size() > i * m_wp + j)
        return NULL;
    
    return m_predictors[i * m_wp + j];
}


//
// GridPredictor<PredictorT>
//


GridPredictor<cv::EM>::GridPredictor(int hp, int wp)
: GridPredictorBase<cv::EM>(hp, wp)
{
    
}

void GridPredictor<cv::EM>::setParameters(GridMat parameters)
{
    m_nmixtures.release();
    m_logthreshold.release();
    
    m_nmixtures.create(m_hp, m_wp, cv::DataType<int>::type);
    m_logthreshold.create(m_hp, m_wp, cv::DataType<int>::type);
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        m_nmixtures.at<int>(i,j) = parameters.at<int>(i,j,0,0);
        this->at(i,j)->set("nclusters", parameters.at<int>(i,j,0,0));
        
        m_logthreshold.at<int>(i,j) = parameters.at<int>(i,j,0,1);
    }
}

void GridPredictor<cv::EM>::setNumOfMixtures(cv::Mat nmixtures)
{
    m_nmixtures = nmixtures;
    
    for (int i = 0; i < nmixtures.rows; i++) for (int j = 0; j < nmixtures.cols; j++)
    {
        this->at(i,j)->set("nclusters", nmixtures.at<int>(i,j));
    }
}

void GridPredictor<cv::EM>::setLoglikelihoodThreshold(cv::Mat loglikes)
{
    m_logthreshold = loglikes;
}

void GridPredictor<cv::EM>::train(GridMat data)
{
    m_data = data;
    
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        this->at(i,j)->train(m_data.at(i,j));
    }
}

/*
 * Returns predictions of the cells, the normalized loglikelihoods [0,1]
 */
void GridPredictor<cv::EM>::predict(GridMat data, GridMat& predictions, GridMat& loglikelihoods, GridMat& distsToMargin)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat& cell = data.at(i,j);
        cv::Mat cellLoglikelihoods (cell.rows, 1, cv::DataType<double>::type);
        
        for (int d = 0; d < cell.rows; d++)
        {
            cv::Vec2d res;
            res = this->at(i,j)->predict(cell.row(d));

            cellLoglikelihoods.at<double>(d,0) = res.val[0];
        }

        cv::Mat normCellLoglikelihoods;
        cv::normalize(cellLoglikelihoods, normCellLoglikelihoods);
        
        cv::Mat cellPredictions;
        cv::threshold(normCellLoglikelihoods, cellPredictions, m_logthreshold.at<float>(i,j), 1, CV_THRESH_BINARY);

        // Center the values around the loglikelihood threshold, so as to have
        // subjects' margin > 0 and objects' margin < 0. And scale to take into
        // accound the variance of the dists' sample
        cv::Mat diffs = normCellLoglikelihoods - m_logthreshold.at<float>(i,j); // center
        cv::Mat powers;
        cv::pow(diffs, 2, powers);
        float scale = sqrt(cv::sum(powers).val[0] / normCellLoglikelihoods.rows);
        cv::Mat cellsDistsToMargin = diffs / scale; // scale
        
        predictions.assign(cellPredictions, i, j);
        loglikelihoods.assign(normCellLoglikelihoods, i, j);
        distsToMargin.assign(cellsDistsToMargin, i, j);
    }
}
