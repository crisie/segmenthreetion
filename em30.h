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
namespace cv
{
class CV_EXPORTS_W EM30 : public Algorithm
{
public:
    // Type of covariation matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};
    
    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    CV_WRAP EM30(int nclusters=EM::DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL,
       const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+
                                                 TermCriteria::EPS,
                                                 EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
    
    virtual ~EM30();
    CV_WRAP virtual void clear();

    CV_WRAP virtual bool train(InputArray samples,
                       OutputArray logLikelihoods=noArray(),
                       OutputArray labels=noArray(),
                       OutputArray probs=noArray());
    
    CV_WRAP virtual bool trainE(InputArray samples,
                        InputArray means0,
                        InputArray covs0=noArray(),
                        InputArray weights0=noArray(),
                        OutputArray logLikelihoods=noArray(),
                        OutputArray labels=noArray(),
                        OutputArray probs=noArray());
    
    CV_WRAP virtual bool trainM(InputArray samples,
                        InputArray probs0,
                        OutputArray logLikelihoods=noArray(),
                        OutputArray labels=noArray(),
                        OutputArray probs=noArray());
    
    CV_WRAP Vec2d predict(InputArray sample,
                OutputArray probs=noArray(), OutputArray loglikelihoods=noArray()) const;

    CV_WRAP bool isTrained() const;

    //AlgorithmInfo* info() const;
    virtual void read(const FileNode& fn);

protected:
    
    virtual void setTrainData(int startStep, const Mat& samples,
                              const Mat* probs0,
                              const Mat* means0,
                              const vector<Mat>* covs0,
                              const Mat* weights0);

    bool doTrain(int startStep,
                 OutputArray logLikelihoods,
                 OutputArray labels,
                 OutputArray probs);
    virtual void eStep();
    virtual void mStep();

    void clusterTrainSamples();
    void decomposeCovs();
    void computeLogWeightDivDet();

    Vec2d computeProbabilities(const Mat& sample, Mat* probs, Mat* logLikelihoods) const;

    // all inner matrices have type CV_64FC1
    CV_PROP_RW int nclusters;
    CV_PROP_RW int covMatType;
    CV_PROP_RW int maxIters;
    CV_PROP_RW double epsilon;

    Mat trainSamples;
    Mat trainProbs;
    Mat trainLogLikelihoods;
    Mat trainLabels;

	Mat maxLogLilelihoodsPerLabel;

    CV_PROP Mat weights;
    CV_PROP Mat means;
    CV_PROP vector<Mat> covs;

    vector<Mat> covsEigenValues;
    vector<Mat> covsRotateMats;
    vector<Mat> invCovsEigenValues;
    Mat logWeightDivDet;
};
} // namespace cv

#endif /* defined(__Segmenthreetion__GridMat__) */
