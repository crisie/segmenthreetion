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


void ThermalBackgroundSubtractor::getMasks(ModalityData& mdInput, ModalityData& mdOutput) {
    
     vector<cv::Mat> masks;
    
    vector<string> calibVarDirs = mdOutput.getCalibVarsDirs();
    
    for(int scene = 0; scene < mdOutput.getNumScenes(); scene++)
    {
        // Create an instance of the registration class
        Registrator minReg;
        
        minReg.loadMinCalibrationVars(calibVarDirs[scene]);
        minReg.toggleUndistortion(false);
        minReg.setUsePrevDepthPoint(true);
    
        //Duplicate depth frames (one duplicated frame per item in that frame, minimum 1 frame -if no item-)
        vector<cv::Mat> replicatedDepthFrames, depthMasksCollection;
        //vector<string> depthIndices;
        vector<int> nMasksPerFrame(mdInput.getSceneSize(scene));
        
        for(unsigned int f = 0; f < mdInput.getSceneSize(scene) ; f++) {
            
            cv::Mat depthFrame = mdInput.getRegFrameInScene(scene,f);
            
            vector<int> uniqueValues;
            findUniqueValues(mdInput.getPredictedMaskInScene(scene, f), uniqueValues);
            
            nMasksPerFrame[f] = 0;
            
            for(unsigned int m = 0; m < uniqueValues.size(); m++) {
                replicatedDepthFrames.push_back(depthFrame);
                depthMasksCollection.push_back(mdInput.getPredictedMaskInScene(scene, f, m));
                nMasksPerFrame[f]++;
            }
            
            if(nMasksPerFrame[f] == 0) {
                replicatedDepthFrames.push_back(depthFrame);
                nMasksPerFrame[f] = 1;
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
                //for (auto c : uniqueValues)
                //    cout << c << ' ';
                
                findUniqueValues(valuedMask, uniqueValues);
                //for (auto c : uniqueValues)
                //    cout << c << ' ';
                
                //Debug -  visualization purposes
                cv::imshow("output", mask);
                cv::waitKey(10);
            }
            
            
            masks.push_back(valuedMask);
            
            //Debug - visualization purposes
            cv::imshow("final mask", valuedMask);
            
            //Debug purposes - show mask unique values
            vector<int> uniqueValues;
            findUniqueValues(valuedMask, uniqueValues);
            //for (auto c : uniqueValues)
            //    cout << c << ' ';
            
            cout << endl;
        }
    
    }
    
    mdOutput.setPredictedMasks(masks);
    
}

void ThermalBackgroundSubtractor::getBoundingRects(ModalityData& mdInput, ModalityData& mdOutput) {
    
    vector<vector<cv::Rect> > boundingRects(mdOutput.getFrames().size());
    
    for(unsigned int f = 0; f < mdOutput.getFrames().size(); f++) {
        
        vector<int> uniqueValuesDepthMask;
        findUniqueValues(mdInput.getPredictedMask(f), uniqueValuesDepthMask);
        uniqueValuesDepthMask.erase(std::remove(uniqueValuesDepthMask.begin(), uniqueValuesDepthMask.end(), 0), uniqueValuesDepthMask.end());
        
        vector<cv::Rect> bbDepth = mdInput.getPredictedBoundingRectsInFrame(f);
        
        for(unsigned int i = 0; i < uniqueValuesDepthMask.size(); i++)
        {
            cv::Rect bigBoundingBox;
            vector<cv::Rect> maskBoundingBoxes;
            this->getMaskBoundingBoxes(mdOutput.getPredictedMask(f,i), maskBoundingBoxes);
            
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
            else if(maskBoundingBoxes.empty() && !bbDepth.empty())
            {
                boundingRects[f].push_back(bbDepth[i]);
            }
            
        }
        
    }
    
    mdOutput.setPredictedBoundingRects(boundingRects);
    
    cout << "Thermal bounding boxes: " << this->countBoundingBoxes(mdOutput.getPredictedBoundingRects()) << endl;
    
    //Debug purposes
    compareNumberBoundingBoxes(mdInput.getPredictedBoundingRects(), mdOutput.getPredictedBoundingRects());
    
    visualizeBoundingRects("Thermal", mdOutput.getFrames(), mdOutput.getPredictedBoundingRects(), true);
    
}

void ThermalBackgroundSubtractor::adaptGroundTruthToReg(ModalityData& md) {
    
    vector<cv::Mat> newThermalGtMasks;
    
    for(int scene = 0; scene < md.getNumScenes(); scene++)
    {

        cv::Mat mask, tempMask(md.getGroundTruthMaskInScene(scene, 0));
        cv::Mat accumulateMask = cv::Mat::zeros(tempMask.size(), tempMask.type());
        
        for(unsigned int f = 0; f < md.getSceneSize(scene); f++)
        {
            cv::Mat auxMask(md.getGroundTruthMaskInScene(scene, f));
            cv::Mat auxThermalGtMask = cv::Mat::zeros(auxMask.size(), CV_8UC1);
            
            threshold(auxMask, auxThermalGtMask, 1, 255, CV_THRESH_BINARY);
            bitwise_or(auxMask,accumulateMask, accumulateMask);
        }
        
        cvtColor(accumulateMask, accumulateMask, CV_RGB2GRAY);
        accumulateMask.convertTo(accumulateMask, CV_8UC1);
        
        for(unsigned int f = 0; f < md.getSceneSize(scene); f++)
        {
            cv::Mat auxMask(md.getGroundTruthMaskInScene(scene, f));
            cv::Mat frameAux = cv::Mat::zeros(auxMask.size(), auxMask.type());
            auxMask.copyTo(frameAux, accumulateMask);
            
            newThermalGtMasks.push_back(frameAux);
        }

    }
    
    md.setGroundTruthMasks(newThermalGtMasks);
    
}

void ThermalBackgroundSubtractor::getRoiTags(ModalityData& mdInput, ModalityData& mdOutput) {
    
    mdOutput.setTags(mdInput.getTags());
}