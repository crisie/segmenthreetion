//
//  em.h
//  segmenthreetion
//
//  Created by Albert Clapés on 05/06/14.
//
//

#ifndef __segmenthreetion__em__
#define __segmenthreetion__em__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "precomp.hpp"

using namespace std;

/****************************************************************************************\
 *                              Expectation - Maximization                                *
 \****************************************************************************************/
namespace cv
{
    class CV_EXPORTS_W EM40 : public cv::EM
    {
    public:
        // Type of covariation matrices
        enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};
        
        // Default parameters
        enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};
        
        // The initial step
        enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};
        
        CV_WRAP EM40(int nclusters=cv::EM::DEFAULT_NCLUSTERS, int covMatType=cv::EM::COV_MAT_DIAGONAL,
                   const cv::TermCriteria& termCrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                                     cv::EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
        
        virtual ~EM40();
        CV_WRAP virtual void clear();
        
        CV_WRAP virtual bool train(cv::InputArray samples,
                                   cv::OutputArray logLikelihoods=cv::noArray(),
                                   cv::OutputArray labels=cv::noArray(),
                                   cv::OutputArray probs=cv::noArray());
        
        CV_WRAP virtual bool trainE(cv::InputArray samples,
                                    cv::InputArray means0,
                                    cv::InputArray covs0=cv::noArray(),
                                    cv::InputArray weights0=cv::noArray(),
                                    cv::OutputArray logLikelihoods=cv::noArray(),
                                    cv::OutputArray labels=cv::noArray(),
                                    cv::OutputArray probs=cv::noArray());
        
        CV_WRAP virtual bool trainM(cv::InputArray samples,
                                    cv::InputArray probs0,
                                    cv::OutputArray logLikelihoods=cv::noArray(),
                                    cv::OutputArray labels=cv::noArray(),
                                    cv::OutputArray probs=cv::noArray());
        
        CV_WRAP cv::Vec3d predict(cv::InputArray sample,
                              cv::OutputArray probs=cv::noArray()) const;
        
        CV_WRAP bool isTrained() const;
        
        cv::AlgorithmInfo* info() const;
        virtual void read(const cv::FileNode& fn);
        
    protected:
        
        virtual void setTrainData(int startStep, const cv::Mat& samples,
                                  const cv::Mat* probs0,
                                  const cv::Mat* means0,
                                  const vector<cv::Mat>* covs0,
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
        
        cv::Vec3d computeProbabilities(const cv::Mat& sample, cv::Mat* probs) const;
        
        // all inner matrices have type CV_64FC1
        CV_PROP_RW int nclusters;
        CV_PROP_RW int covMatType;
        CV_PROP_RW int maxIters;
        CV_PROP_RW double epsilon;
        
        cv::Mat trainSamples;
        cv::Mat trainProbs;
        cv::Mat trainLogLikelihoods;
        cv::Mat trainLabels;
        
        CV_PROP cv::Mat weights;
        CV_PROP cv::Mat means;
        CV_PROP vector<cv::Mat> covs;
        
        vector<cv::Mat> covsEigenValues;
        vector<cv::Mat> covsRotateMats;
        vector<cv::Mat> invCovsEigenValues;
        cv::Mat logWeightDivDet;
    };
    
    CV_INIT_ALGORITHM(EM40, "StatModel.EM",
                      obj.info()->addParam(obj, "nclusters", obj.nclusters);
                      obj.info()->addParam(obj, "covMatType", obj.covMatType);
                      obj.info()->addParam(obj, "maxIters", obj.maxIters);
                      obj.info()->addParam(obj, "epsilon", obj.epsilon);
                      obj.info()->addParam(obj, "weights", obj.weights, true);
                      obj.info()->addParam(obj, "means", obj.means, true);
                      obj.info()->addParam(obj, "covs", obj.covs, true));
    
    bool initModule_ml(void)
    {
        Ptr<Algorithm> em = createEM40();
        return em->info() != 0;
    }
    
} // namespace cvx

#endif /* defined(__segmenthreetion__em__) */
