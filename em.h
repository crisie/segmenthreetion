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
        
        CV_WRAP EM40(int nclusters=cv::EM::DEFAULT_NCLUSTERS, int covMatType=cv::EM::COV_MAT_DIAGONAL,
                   const cv::TermCriteria& termCrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                                     cv::EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
        
        CV_WRAP cv::Vec3d predict(cv::InputArray sample,
                              cv::OutputArray probs=cv::noArray()) const;
        
    protected:
        
        virtual void eStep();
        
        cv::Vec3d computeProbabilities(const cv::Mat& sample, cv::Mat* probs) const;
    };
} // namespace cv

#endif /* defined(__segmenthreetion__em__) */
