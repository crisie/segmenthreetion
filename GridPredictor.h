//
//  GridPredictor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#ifndef __segmenthreetion__GridPredictor__
#define __segmenthreetion__GridPredictor__


#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "GridMat.h"

using namespace std;

template<typename PredictorT>
class GridPredictorBase
{
public:
    GridPredictorBase();
    
    void setData(GridMat<float> data, cv::Mat categories);
    
    PredictorT& at(unsigned int i, unsigned int j);

protected:
    GridMat<float> m_data;
    cv::Mat m_categories;
    
    unsigned int m_hp, m_wp;
    
    vector<PredictorT> m_predictors;
};


template<typename PredictorT>
class GridPredictor : public GridPredictorBase<PredictorT>
{
    GridPredictor();
    
    void setData(GridMat<float> data, cv::Mat categories);
    
    PredictorT& t(unsigned int i, unsigned int j);
};


template<>
class GridPredictor<cv::EM> : public GridPredictorBase<cv::EM>
{
public:
    GridPredictor();
    
    void setData(GridMat<float> data, cv::Mat categories);
    void setParameters(GridMat<int> parameters);
    void setNumOfMixtures(int m);
    void setLoglikelihoodThreshold(int t);
    
    cv::EM& at(unsigned int i, unsigned int j);
    
    void train();
    void predict(GridMat<float> data, GridMat<int>& predictions, GridMat<int>& loglikelihoods);
    
private:
    int m_nmixtures;
    int m_logthreshold;
    GridMat<int> m_Parameters;
};



#endif /* defined(__segmenthreetion__GridPredictor__) */
