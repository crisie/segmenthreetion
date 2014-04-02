//
//  ColorParametrization.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__ColorParametrization__
#define __Segmenthreetion__ColorParametrization__

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class ColorParametrization
{
public:
    ColorParametrization() {};
    
    int gridSizeX; //= 64;
    int gridSizeY; // = 128;
    
    bool signedGradient;
    
    int winSizeX; // =128;
    int winSizeY; // = 64;
    int blockSizeX; // = 16;
    int blockSizeY; // = 16;
    int blockStrideX; // = 8;
    int blockStrideY; // = 8;
    int cellSizeX; // = 8;
    int cellSizeY; // = 8;
    int nbins; //= 9;
    int derivAper; // = 0;
    int winSigma; // = -1;
    int histNormType; // = 0;
    float L2HysThresh; // = 0.2;
    int gammaCorrection; // = 0;
    int nLevels; // = 64;
    int hogbins; // = 288
    
};

#endif /* defined(__Segmenthreetion__ColorParametrization__) */