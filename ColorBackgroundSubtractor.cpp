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


void ColorBackgroundSubtractor::getMasks(ModalityData& mdOutput, ModalityData& mdInput) {
    
    mdOutput.setMasks(mdInput.getMasks());
    
}

void ColorBackgroundSubtractor::getBoundingRects(ModalityData& mdOutput, ModalityData& mdInput) {

    mdOutput.setBoundingRects(mdInput.getBoundingRects());
    
    cout << "Color bounding boxes: " << this->countBoundingBoxes(mdOutput.getBoundingRects()) << endl;

}

void ColorBackgroundSubtractor::adaptGroundTruthToReg(ModalityData& mdOutput, ModalityData& mdInput) {
    
    mdOutput.setGroundTruthMasks(mdInput.getGroundTruthMasks());
    
}

void ColorBackgroundSubtractor::getRoiTags(ModalityData& mdOutput, ModalityData& mdInput) {
    
    mdOutput.setTags(mdInput.getTags());
}

void ColorBackgroundSubtractor::getGroundTruthBoundingRects(ModalityData& mdOutput, ModalityData& mdInput) {
    
    mdOutput.setGroundTruthBoundingRects(mdInput.getGroundTruthBoundingRects());
    
}