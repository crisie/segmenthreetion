//
//  ForegroundParametrization.hpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 06/03/14.
//
//

#ifndef segmenthreetion_ForegroundParametrization_hpp
#define segmenthreetion_ForegroundParametrization_hpp

class ForegroundParametrization
{
public:
    ForegroundParametrization() {}
    
    std::vector<int> numFramesToLearn;
    
    float boundingBoxMinArea;
    float otsuMinArea;
    float otsuMinVariance1;
    float otsuMinVariance2;
};



#endif
