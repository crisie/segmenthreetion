//
//  DepthParametrization.cpp
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#include "DepthParametrization.h"

DepthParametrization::DepthParametrization()
{}

DepthParametrization::DepthParametrization(int thetaBins, int phiBins, float normalsRadius)
    : thetaBins(thetaBins), phiBins(phiBins), normalsRadius(normalsRadius)
{}