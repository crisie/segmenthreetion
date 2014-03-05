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
GridPredictorBase<PredictorT>::GridPredictorBase()
{
    
}

template<typename PredictorT>
void GridPredictorBase<PredictorT>::setData(GridMat<float> data, cv::Mat categories)
{
    m_data = data;
    m_categories = categories;
    m_hp = data.crows();
    m_wp = data.ccols();
    m_predictors.resize(data.crows() * data.ccols());
}

template<typename PredictorT>
PredictorT& GridPredictorBase<PredictorT>::at(unsigned int i, unsigned int j)
{
    return m_predictors[i * m_wp + j];
}


//
// GridPredictor<PredictorT>
//

template<typename PredictorT>
GridPredictor<PredictorT>::GridPredictor()
: GridPredictorBase<PredictorT>()
{
    
}

template<typename PredictorT>
void GridPredictor<PredictorT>::setData(GridMat<float> data, cv::Mat categories)
{
    GridPredictorBase<PredictorT>::setData(data, categories);
}

template<typename PredictorT>
PredictorT& GridPredictor<PredictorT>::at(unsigned int i, unsigned int j)
{
    return GridPredictorBase<PredictorT>::getPredictor(i, j);
}


//
// GridPredictor<cv::EM>
//

GridPredictor<cv::EM>::GridPredictor()
: GridPredictorBase<cv::EM>()
{
    
}

void GridPredictor<cv::EM>::setData(GridMat<float> data, cv::Mat categories)
{
    GridPredictorBase<cv::EM>::setData(data, categories);
}

void GridPredictor<cv::EM>::setParameters(GridMat<int> parameters)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        this->at(i,j).set("nclusters", parameters.at(i,j,0,0));
    }
}

cv::EM& GridPredictor<cv::EM>::getPredictor(unsigned int i, unsigned int j)
{
    return GridPredictorBase<cv::EM>::at(i, j);
}

void GridPredictor<cv::EM>::setNumOfMixtures(int m)
{
    m_nmixtures = m;
}

void GridPredictor<cv::EM>::setLoglikelihoodThreshold(int t)
{
    m_logthreshold = t;
}

void GridPredictor<cv::EM>::train()
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        this->at(i,j).train(m_data.get(i,j));
    }
}

void GridPredictor<cv::EM>::predict(GridMat<float> data, GridMat<int>& predictions, GridMat<int>& loglikelihoods)
{
    for (int i = 0; i < m_hp; i++) for (int j = 0; j < m_wp; j++)
    {
        cv::Mat& cell = data.at(i,j);
        cv::Mat cellPredictions (cell.rows, 1, cv::DataType<int>::type);
        cv::Mat cellLoglikelihoods (cell.rows, 1, cv::DataType<int>::type);
        
        for (int d = 0; d < cell.rows; d++)
        {
            cv::Vec2d res;
            res = this->at(i,j).predict(cell.row(d));
            
            cellPredictions.at<int>(d,0) = (res[0] > m_logthreshold);
            cellLoglikelihoods.at<int>(d,0) = res[0];
        }

        predictions.set(cellPredictions, i, j);
        loglikelihoods.set(cellLoglikelihoods, i, j);
    }
}


// Explicit template instanciation (to avoid linking errors)
template typename GridPredictorBase<cv::EM>;
template typename GridPredictor<cv::EM>;