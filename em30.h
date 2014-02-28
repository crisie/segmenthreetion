//
//  GridMat.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 21/04/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__EM30__
#define __Segmenthreetion__EM30__

#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/

class CV_EXPORTS_W EM30 : public cv::Algorithm
{
public:
    // Type of covariation cv::Matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};
    
    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    CV_WRAP EM30(int nclusters=cv::EM::DEFAULT_NCLUSTERS, int covMatType=cv::EM::COV_MAT_DIAGONAL,
                 const cv::TermCriteria& termCrit = cv::TermCriteria(cv::TermCriteria::COUNT+
                                                                    cv::TermCriteria::EPS,
                                                                     cv::EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
    
    virtual ~EM30();
    CV_WRAP virtual void clear();

    CV_WRAP virtual bool train(cv::InputArray samples,
                               cv::OutputArray logLikelihoods=cv::noArray(),
                               cv::OutputArray labels=cv::noArray(),
                               cv::OutputArray probs=cv::noArray());
    
    CV_WRAP virtual bool trainE(cv::InputArray samples,
                        cv::InputArray means0,
                                cv::InputArray covs0=cv::noArray(),
                        cv::InputArray weights0= cv::noArray(),
                        cv::OutputArray logLikelihoods= cv::noArray(),
                        cv::OutputArray labels= cv::noArray(),
                        cv::OutputArray probs= cv::noArray());
    
    CV_WRAP virtual bool trainM(cv::InputArray samples,
                        cv::InputArray probs0,
                        cv::OutputArray logLikelihoods= cv::noArray(),
                        cv::OutputArray labels= cv::noArray(),
                        cv::OutputArray probs= cv::noArray());
    
    CV_WRAP cv::Vec2d predict(cv::InputArray sample,
                              cv::OutputArray probs= cv::noArray(), cv::OutputArray loglikelihoods=cv::noArray()) const;

    CV_WRAP bool isTrained() const;

    //AlgorithmInfo* info() const;
    virtual void read(const cv::FileNode& fn);

protected:
    
    virtual void setTrainData(int startStep, const cv::Mat& samples,
                              const cv::Mat* probs0,
                              const cv::Mat* means0,
                              const std::vector<cv::Mat>* covs0,
                              const cv::Mat* weights0);

    bool doTrain(int startStep,
                 cv::OutputArray logLikelihoods,
                 cv::OutputArray labels,
                 cv::OutputArray probs);
    virtual void eStep();
    virtual void mStep();

    void clusterTrainSamples();
    void decomposeCovs();
    void computeLogWeightDivDet();

    cv::Vec2d computeProbabilities(const cv::Mat& sample, cv::Mat* probs, cv::Mat* logLikelihoods) const;

    // all inner cv::Matrices have type CV_64FC1
    CV_PROP_RW int nclusters;
    CV_PROP_RW int covMatType;
    CV_PROP_RW int maxIters;
    CV_PROP_RW double epsilon;

    cv::Mat trainSamples;
    cv::Mat trainProbs;
    cv::Mat trainLogLikelihoods;
    cv::Mat trainLabels;

    cv::Mat maxLogLilelihoodsPerLabel;

    CV_PROP cv::Mat weights;
    CV_PROP cv::Mat means;
    CV_PROP std::vector<cv::Mat> covs;

    std::vector<cv::Mat> covsEigenValues;
    std::vector<cv::Mat> covsRotateMats;
    std::vector<cv::Mat> invCovsEigenValues;
    cv::Mat logWeightDivDet;
};

#endif /* defined(__Segmenthreetion__GridMat__) */
