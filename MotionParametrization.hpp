/*
 * MotionParametrization.h
 *
 *  Created on: 02/06/2013
 *      Author: Cristina
 */

#ifndef  __Segmenthreetion__MotionParametrization__
#define  __Segmenthreetion__MotionParametrization__

#include <iostream>

class MotionParametrization
{
public:
    MotionParametrization() {}
    
    int hoofbins;      // Bins in the histogram of oriented optical flows
    
    //Farneback optical flow parameters:
	double pyr_scale;// = 0.5;
	int levels;// = 3;
	int winsize;// = 15;
	int iterations;// = 3;
	int poly_n;// = 5;
	double poly_sigma;// = 1.2;
	int flags;// = 0;
};


#endif  /* defined(__Segmenthreetion__DepthParametrization__) */
