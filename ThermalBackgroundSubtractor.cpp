//
//  ThermalBackgroundSubtractor.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 07/03/14.
//
//

#include "registrator.h"
#include "BackgroundSubtractor.h"
#include "ThermalBackgroundSubtractor.h"

#include "StatTools.h"
#include "DebugTools.h"

#include <opencv2/opencv.hpp>


ThermalBackgroundSubtractor::ThermalBackgroundSubtractor()
    : BackgroundSubtractor()
{ }


void ThermalBackgroundSubtractor::getMasks(ModalityData& mdOutput, ModalityData& mdInput) {
    
     vector<cv::Mat> masks;

    //TODO: revise getters used!
    
    for(int scene = 0; scene < mdOutput.getNumScenes(); scene++)
    {
        // Create an instance of the registration class
        Registrator minReg;
        
        minReg.loadMinCalibrationVars(mdOutput.getCalibVarsDirs()[scene]);
        minReg.setUsePrevDepthPoint(true);
    
        //Duplicate depth frames (one duplicated frame per item in that frame, minimum 1 frame -if no item-)
        vector<cv::Mat> depthFrames, replicatedDepthFrames, depthMasksCollection;
        //vector<string> depthIndices;
        vector<int> nMasksPerFrame;
        
        depthFrames = mdOutput.getFramesInScene(scene);
        
        for(unsigned int f = 0; f < mdInput.getFramesInScene(scene).size() ; f++) {
            
            cv::Mat depthFrame = mdInput.getFrameInScene(scene,f);
            
            vector<int> uniqueValues;
            findUniqueValues(mdOutput.getPredictedMaskInScene(scene, f), uniqueValues);
            
            replicatedDepthFrames.push_back(depthFrame);
            nMasksPerFrame[f] = 1;
            
            for(unsigned int m = 1; m < uniqueValues.size(); m++) {
                replicatedDepthFrames.push_back(depthFrame);
                depthMasksCollection.push_back(mdOutput.getPredictedMaskInScene(scene, f, m));
                nMasksPerFrame[f]++;
            }
            
        }
        
        //Register individual depth/rgb masks to individual thermal masks
        vector<cv::Mat> thermalMasksCollection;
        minReg.loadRegSaveContours(depthMasksCollection, depthMasksCollection, thermalMasksCollection, replicatedDepthFrames);
        
        int index = 0;
        for(unsigned int f = 0; f < nMasksPerFrame.size(); f++) {
            
            int nItem = 0;
            cv::Mat valuedMask = cv::Mat::zeros(thermalMasksCollection[index].size(), CV_8UC1);
            
            while (nItem < nMasksPerFrame[f])
            {
                cv::Mat mask = cv::Mat::zeros(thermalMasksCollection[index].size(), CV_8UC1);
                thermalMasksCollection[index].copyTo(mask);
                
                this->changePixelValue(mask, nItem);
                
                add(valuedMask, mask, valuedMask);
                this->changePixelValue(valuedMask, 255, nItem);
                
                nItem++;
                index++;
                
                //Debug purposes - show mask unique values
                vector<int> uniqueValues;
                findUniqueValues(mask, uniqueValues);
                for (auto c : uniqueValues)
                    cout << c << ' ';
                
                findUniqueValues(valuedMask, uniqueValues);
                for (auto c : uniqueValues)
                    cout << c << ' ';
            }
            
            masks.push_back(valuedMask);
            
            //Debug purposes - show mask unique values
            vector<int> uniqueValues;
            findUniqueValues(valuedMask, uniqueValues);
            for (auto c : uniqueValues)
                cout << c << ' ';
            
            cout << endl;
        }
    
    }
    
    //TODO: save somehow masks in folder
    
    mdOutput.setPredictedMasks(masks);
    
}

void ThermalBackgroundSubtractor::getBoundingRects(ModalityData& mdOutput, ModalityData& mdInput) {
    
    vector<vector<cv::Rect> > boundingRects(mdInput.getFrames().size());
    
    for(unsigned int f = 0; f < mdInput.getFrames().size(); f++) {
        
        vector<int> uniqueValuesDepthMask;
        findUniqueValues(mdOutput.getPredictedMask(f), uniqueValuesDepthMask);
        
        for(unsigned int i = 0; i < uniqueValuesDepthMask.size(); i++)
        {
            cv::Rect bigBoundingBox;
            vector<cv::Rect> maskBoundingBoxes;
            this->getMaskBoundingBoxes(mdInput.getPredictedMask(f), maskBoundingBoxes);
            
            if(!maskBoundingBoxes.empty())
            {
                this->getMaximalBoundingBox(maskBoundingBoxes, bigBoundingBox);
                
                if(!this->checkMinimumBoundingBoxes(bigBoundingBox, 4))
                {
                    boundingRects[f].push_back(getMinimumBoundingBox(bigBoundingBox, 4));
                }
                else
                {
                    boundingRects[f].push_back(bigBoundingBox);
                }
            }
            else if(maskBoundingBoxes.empty() && !mdOutput.getPredictedBoundingRectsInFrame(f).empty())
            {
                boundingRects[f].push_back(mdOutput.getPredictedBoundingRectsInFrame(f)[i]);
            }
            
        }
        
    }
    
    mdOutput.setPredictedBoundingRects(boundingRects);
    
    cout << "Thermal bounding boxes: " << this->countBoundingBoxes(mdOutput.getPredictedBoundingRects()) << endl;
    
    //Debug purposes
    compareNumberBoundingBoxes(mdInput.getPredictedBoundingRects(), mdOutput.getPredictedBoundingRects());
    
}

void ThermalBackgroundSubtractor::adaptGroundTruthToReg(ModalityData& md) {
    
    vector<cv::Mat> newThermalGtMasks;
    
    for(int scene = 0; scene < md.getNumScenes(); scene++)
    {
        vector<cv::Mat> thermalGtMasks = md.getGroundTruthMasksInScene(scene);

        cv::Mat mask;
        cv::Mat accumulateMask = cv::Mat::zeros(thermalGtMasks[0].size(), thermalGtMasks[0].type());
        
        for(unsigned int f = 0; f < thermalGtMasks.size(); f++)
        {
            cv::Mat auxThermalGtMask = cv::Mat::zeros(thermalGtMasks[f].size(), CV_8UC1);
            
            threshold(thermalGtMasks[f], auxThermalGtMask, 1, 255, CV_THRESH_BINARY);
            bitwise_or(thermalGtMasks[f],accumulateMask, accumulateMask);
        }
        
        cvtColor(accumulateMask, accumulateMask, CV_RGB2GRAY);
        accumulateMask.convertTo(accumulateMask, CV_8UC1);
        
        for(unsigned int f = 0; f < thermalGtMasks.size(); f++)
        {
            cv::Mat frameAux = cv::Mat::zeros(thermalGtMasks[f].size(), thermalGtMasks[f].type());
            thermalGtMasks[f].copyTo(frameAux, accumulateMask);
            
            newThermalGtMasks.push_back(frameAux);
        }
        
    }
    
    md.setGroundTruthMasks(newThermalGtMasks);
    
    //TODO: save in dir somehow
}