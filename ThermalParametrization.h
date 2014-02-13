//
//  ThermalParametrization.h
//  Segmenthreetion
//
//  Created by Albert Clapés on 24/05/13.
//  Copyright (c) 2013 Albert Clapés. All rights reserved.
//

#ifndef __Segmenthreetion__ThermalParametrization__
#define __Segmenthreetion__ThermalParametrization__

#include <iostream>


class ThermalParametrization
{
public:
    ThermalParametrization();
    ThermalParametrization(int ibins, int oribins);
    
    int ibins;      // Bins in the intensity histograms
    int oribins;    // Bins in the gradient orientation histograms
};

#endif /* defined(__Segmenthreetion__ThermalParametrization__) */
