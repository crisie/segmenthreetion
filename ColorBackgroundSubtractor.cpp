//
//  ColorBackgroundSubtractor.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 07/03/14.
//
//

#include "BackgroundSubtractor.h"
#include "ColorBackgroundSubtractor.h"

#include <opencv2/opencv.hpp>


ColorBackgroundSubtractor::ColorBackgroundSubtractor()
: BackgroundSubtractor()
{ }


/*
void ColorBackgroundSubtractor::setMasksOffset(unsigned char masksOffset) {
    m_masksOffset = masksOffset;
}
 */


void ColorBackgroundSubtractor::getMasks(ModalityData& mdInput, ModalityData& mdOutput) {
    
    mdOutput.setPredictedMasks(mdInput.getPredictedMasks());
    
}

void ColorBackgroundSubtractor::getBoundingRects(ModalityData& mdInput, ModalityData& mdOutput) {

    mdOutput.setPredictedBoundingRects(mdInput.getPredictedBoundingRects());
    
    cout << "Color bounding boxes: " << this->countBoundingBoxes(mdOutput.getPredictedBoundingRects()) << endl;

}

void ColorBackgroundSubtractor::adaptGroundTruthToReg(ModalityData& mdInput, ModalityData& mdOutput) {
    
    mdOutput.setGroundTruthMasks(mdInput.getGroundTruthMasks());
    
}

void ColorBackgroundSubtractor::getRoiTags(ModalityData& mdInput, ModalityData& mdOutput) {
    
    mdOutput.setTags(mdInput.getTags());
}

void ColorBackgroundSubtractor::getGroundTruthBoundingRects(ModalityData& mdInput, ModalityData& mdOutput) {
    
    mdOutput.setGroundTruthBoundingRects(mdInput.getGroundTruthBoundingRects());
    
}