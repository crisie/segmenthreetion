//
//  StatTools.h
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#ifndef __segmenthreetion__StatTools__
#define __segmenthreetion__StatTools__

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

double phi(double x);
void histogram(cv::Mat mat, int nbins, cv::Mat & hist);

#endif /* defined(__segmenthreetion__StatTools__) */
