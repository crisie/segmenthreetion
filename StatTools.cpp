//
//  StatTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "StatTools.h"

#include <math.h>

double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    
    return 0.5*(1.0 + sign*y);
}

void histogram(cv::Mat mat, int nbins, cv::Mat & hist)
{
    double min, max;
    cv::minMaxIdx(mat, &min, &max);
    
    if (nbins > (max-min+1))
        return;
    
    // Create an histogram for the cell region of blurred intensity values
    int histSize[] = { (int) nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float tranges[] = { min, max }; // thermal intensity values range: [0, 256)
    const float* ranges[] = { tranges };
    
    cv::calcHist(&mat, 1, channels, cv::noArray(), hist, 1, histSize, ranges, true, false);
}