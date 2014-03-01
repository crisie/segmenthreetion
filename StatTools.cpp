//
//  StatTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "StatTools.h"

#include <math.h>


/**
 * Builds an histogram of the values contained in a vector (or matrix)
 */
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

/**
 * Create a column vector containing the numbers in the interval [a,b] shuffled randomly
 */
cv::Mat shuffledVector(int a, int b, cv::RNG randGen)
{
    cv::Mat vec (b-a+1, 1, cv::DataType<int>::type);
    for (int i = a; i <= b; i++)
    {
        vec.at<int>(i-a, 0) = i;
    }
    
    randShuffle(vec, 1, &randGen);
    
    return vec;
}

/**
 * Create a column vector containing the numbers in the interval [0,n) shuffled randomly
 */
cv::Mat shuffledVector(int n, cv::RNG randGen)
{
    return shuffledVector(0, n-1, randGen);
}

/**
 * Create a vector of labels representing the k folds of n elements
 */
void cvpartition(int n, int k, int seed, cv::Mat& labels)
{
    int foldElems = floor(n/k);
    int extraElems = n - (k * floor(n/k));

    cv::Mat indices = shuffledVector(n, cv::RNG(seed));
    
    labels.release();
    labels.create(indices.rows, indices.cols, cv::DataType<int>::type);

    unsigned int c = 0;
    unsigned int i, j;
    for (i = 0; i < k; i++)
    {
        int ifoldElems = (i < extraElems) ? (foldElems + 1) : foldElems;
        for (j = 0; j < ifoldElems; j++)
        {
            labels.at<int>(indices.at<int>(c+j)) = i;
        }
        
        c += ifoldElems;
    }
}

/**
 * Mathematical function approximating a Gaussian function
 */
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