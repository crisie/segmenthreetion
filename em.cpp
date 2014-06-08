/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                        Intel License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright( C) 2000, Intel Corporation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of Intel Corporation may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 //(including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort(including negligence or otherwise) arising in any way out of
 // the use of this software, even ifadvised of the possibility of such damage.
 //
 //M*/


//
//  em.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 05/06/14.
//
//


#include "em.h"
#include "precomp.hpp"

using namespace std;
using namespace cv;

namespace cv
{
    EM40::EM40(int _nclusters, int _covMatType, const TermCriteria& _termCrit)
    : cv::EM(_nclusters, _covMatType, _termCrit)
    {
    }
    
    Vec3d EM40::predict(InputArray _sample, OutputArray _probs) const
    {
        Mat sample = _sample.getMat();
        CV_Assert(isTrained());
        
        CV_Assert(!sample.empty());
        if(sample.type() != CV_64FC1)
        {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }
        sample.reshape(1, 1);
        
        Mat probs;
        if( _probs.needed() )
        {
            _probs.create(1, nclusters, CV_64FC1);
            probs = _probs.getMat();
        }
        
        return computeProbabilities(sample, !probs.empty() ? &probs : 0);
    }
  
    Vec3d EM40::computeProbabilities(const Mat& sample, Mat* probs) const
    {
        // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
        // q = arg(max_k(L_ik))
        // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
        // see Alex Smola's blog http://blog.smola.org/page/2 for
        // details on the log-sum-exp trick
        
        CV_Assert(!means.empty());
        CV_Assert(sample.type() == CV_64FC1);
        CV_Assert(sample.rows == 1);
        CV_Assert(sample.cols == means.cols);
        
        int dim = sample.cols;
        
        Mat L(1, nclusters, CV_64FC1);
        int label = 0;
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            const Mat centeredSample = sample - means.row(clusterIndex);
            
            Mat rotatedCenteredSample = covMatType != EM40::COV_MAT_GENERIC ?
            centeredSample : centeredSample * covsRotateMats[clusterIndex];
            
            double Lval = 0;
            for(int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != EM40::COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;
            
            if(L.at<double>(clusterIndex) > L.at<double>(label))
                label = clusterIndex;
        }
        
        double maxLVal = L.at<double>(label);
        Mat expL_Lmax = L; // exp(L_ij - L_iq)
        for(int i = 0; i < L.cols; i++)
            expL_Lmax.at<double>(i) = std::exp(L.at<double>(i) - maxLVal);
        double expDiffSum = sum(expL_Lmax)[0]; // sum_j(exp(L_ij - L_iq))
        
        if(probs)
        {
            probs->create(1, nclusters, CV_64FC1);
            double factor = 1./expDiffSum;
            expL_Lmax *= factor;
            expL_Lmax.copyTo(*probs);
        }
        
        Vec3d res;
        res[0] = std::log(expDiffSum)  + maxLVal - 0.5 * dim * CV_LOG2PI;
        res[1] = maxLVal - 0.5 * dim * CV_LOG2PI;
        res[2] = label;
        
        return res;
    }
    
    void EM40::eStep()
    {
        // Compute probs_ik from means_k, covs_k and weights_k.
        trainProbs.create(trainSamples.rows, nclusters, CV_64FC1);
        trainLabels.create(trainSamples.rows, 1, CV_32SC1);
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);
        
        computeLogWeightDivDet();
        
        CV_DbgAssert(trainSamples.type() == CV_64FC1);
        CV_DbgAssert(means.type() == CV_64FC1);
        
        for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex);
            Vec3d res = computeProbabilities(trainSamples.row(sampleIndex), &sampleProbs);
            trainLogLikelihoods.at<double>(sampleIndex) = res[0];
            trainLabels.at<int>(sampleIndex) = static_cast<int>(res[2]);
        }
    }
    
} // namespace cvx

/* End of file. */
