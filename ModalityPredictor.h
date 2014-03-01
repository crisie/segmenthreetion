//
//  ModalityPredictor.h
//  segmenthreetion
//
//  Created by Albert Clapés on 02/03/14.
//
//

#ifndef __segmenthreetion__ModalityPredictor__
#define __segmenthreetion__ModalityPredictor__

#include <iostream>

class ModalityPredictor
{
public:
    ModalityPredictor();
    
    void setModelSelection(int k, cv::Mat params);
    void setValidationParams(int k);
    
    
};

#endif /* defined(__segmenthreetion__ModalityPredictor__) */
