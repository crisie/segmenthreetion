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

#include "GridMat.h"

// Builds an histogram of the values contained in a vector (or matrix)
void histogram(cv::Mat mat, int nbins, cv::Mat & hist);

// Create a column vector containing the numbers in the interval [a,b] shuffled randomly
cv::Mat shuffledVector(int a, int b, cv::RNG randGen);

// Create a column vector containing the numbers in the interval [0,n) shuffled randomly
cv::Mat shuffledVector(int n, cv::RNG randGen);

// Create a vector of labels representing the k folds of n elements
void cvpartition(int n, int k, int seed, cv::Mat& partitions);
void cvpartition(cv::Mat labels, int k, int seed, cv::Mat& partitions); // stratified

// Mathematical function approximating a Gaussian function
double phi(double x);

// Sort a vector by unique values
void uniqueSortValues(vector<int> & values);

// Find unique values of a Mat and returns them sorted
void findUniqueValues(cv::Mat image, vector<int> & values);

// Find unique values of a vector and returns them sorted
void findUniqueValues(vector<int> v, vector<int> & values);

#endif /* defined(__segmenthreetion__StatTools__) */
