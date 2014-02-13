//
//  DepthParametrization.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__DepthParametrization__
#define __Segmenthreetion__DepthParametrization__

#include <iostream>

class DepthParametrization
{
public:
	DepthParametrization();
    DepthParametrization(int thetaBins, int phiBins, float normalsRadius);

	int thetaBins;
    int phiBins;
    
    float normalsRadius;
};

#endif /* defined(__Segmenthreetion__DepthParametrization__) */
