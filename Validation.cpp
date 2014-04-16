//
//  Validation.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 21/03/14.
//
//

#include "Validation.h"
#include "StatTools.h"




using namespace std;

Validation::Validation()
{ }

void Validation::getOverlap(ModalityData& md, vector<float> dcRange, cv::Mat& overlapIDs) {
    
    cv::Mat newOverlapIDs(cvSize(md.getPredictedMasks().size(), dcRange.size()+1), CV_8UC1);
    this->getOverlap(md.getPredictedMasks(), md.getGroundTruthMasks(), dcRange, newOverlapIDs);
    cv::hconcat(overlapIDs, newOverlapIDs, overlapIDs);
    
}

void Validation::getOverlap(vector<cv::Mat> predictedMasks, vector<cv::Mat> gtMasks, vector<float> dcRange, cv::Mat& overlapIDs) {
    
    //cv::resize(overlapIDs, overlapIDs, cvSize(predictedMasks.size(), dcRange.size()+1));
    int idx = 0;
    for(int f = 0; f < predictedMasks.size(); f++)
    {
        if(cv::countNonZero(predictedMasks[f]) > 0 || cv::countNonZero(gtMasks[f]) > 0)
        {
            overlapIDs.at<float>(0, idx) = getMaskOverlap(predictedMasks[f], gtMasks[f], cv::Mat());
            
            for(int dc = 0; dc < dcRange.size(); dc++)
            {
                cv::Mat dontCare;
                createDontCareRegion(gtMasks[f], dontCare, dcRange[dc]);
                threshold(dontCare, dontCare, 128, 255, CV_THRESH_BINARY);
                
                overlapIDs.at<float>(dc+1, idx) = getMaskOverlap(predictedMasks[f], gtMasks[f], dontCare);
            }
            idx++;
        }
    }
    
    //TODO: Check if all iDs are finite or treat it when computing the mean (outside this function ?)
    
}

/*
 * Get overlap value between predicted mask and ground truth mask based on Jaccard Similarity/Index
 */
float Validation::getMaskOverlap(cv::Mat predictedMask, cv::Mat gtMask, cv::Mat dontCareRegion)
{
    bool useDontCare = false;
    if(!dontCareRegion.empty()) useDontCare = true;
    
    double overlap = 0.0;
    int nBB = 0;
    vector<double> overlapIDs;
    
    vector<int> gtMaskPersonID;
    findUniqueValues(gtMask, gtMaskPersonID);
    gtMaskPersonID.erase(std::remove(gtMaskPersonID.begin(), gtMaskPersonID.end(), 0), gtMaskPersonID.end());
    
    //labeled_result_mask: we already have one label per blob (200, 201, 202....)
    vector<int> predictedMaskPersonID;
    findUniqueValues(predictedMask, predictedMaskPersonID);
    predictedMaskPersonID.erase(std::remove(predictedMaskPersonID.begin(), predictedMaskPersonID.end(), 0), predictedMaskPersonID.end());
    
    cv::Mat labeledGtMask = cv::Mat::zeros(gtMask.size(), CV_8UC1);
    vector<vector<cv::Point> > contours;
    findContours(gtMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for(int c = 0; c < contours.size(); c++)
    {
        drawContours(labeledGtMask, contours, c, c, CV_FILLED, 8, vector<cv::Vec4i>());
    }
    
    vector<bool> gtMaskUsedIDs(gtMaskPersonID.size());
    std::fill(gtMaskPersonID.begin(), gtMaskPersonID.end(), false);
    
    vector<bool> prMaskUsedIDs(predictedMaskPersonID.size());
    std::fill(prMaskUsedIDs.begin(), prMaskUsedIDs.end(), false);
    
    if(!predictedMaskPersonID.empty())
    {
        std::vector<int>::iterator gtID = gtMaskPersonID.begin();
        while (gtID != gtMaskPersonID.end())
        {
            int id  = *gtID;
            cv::Mat person = (gtMask == id);
            threshold(person, person, 128, 255, CV_THRESH_BINARY);
            
            vector<int> personsInBlob;
            personsInBlob.push_back(id);
            
            vector<int> labelsOverlap;
            cv::Mat resultRegionMask;

            bool allChecked = false;
            while(!allChecked)
            {
                
                vector<int> personsGt;
                personsGt.insert(personsGt.end(), personsInBlob.begin(), personsInBlob.end());
                
                /* 1. Check if there is more than one person inside a region.
                 *    Check to which blob id corresponds that person id.
                 */
                bool gtChecked = false;
                while(!gtChecked)
                {
                    vector<int> regionIDs;
                    cv::Mat maskedLabeledGtMask;
                    labeledGtMask.copyTo(maskedLabeledGtMask, person);
                    findUniqueValues(maskedLabeledGtMask, regionIDs);
                    
                    for(int i = 0 ; i < regionIDs.size(); i++)
                    {
                        vector<int> uniqueIDs;
                        findUniqueValues(labeledGtMask == regionIDs[i], uniqueIDs);
                        personsInBlob.insert(personsInBlob.end(), uniqueIDs.begin(), uniqueIDs.end());
                    }
                    
                    findUniqueValues(personsInBlob, personsInBlob);
                    
                    if(personsGt.size() != personsInBlob.size())
                    {
                        person.release();
                        for(int i = 0; i < personsInBlob.size(); i++)
                        {
                            add(person, gtMask == personsInBlob[i], person);
                        }
                        threshold(person, person, 128, 255, CV_THRESH_BINARY);
                        
                        personsGt.clear();
                        personsGt.insert(personsGt.end(), personsInBlob.begin(), personsInBlob.end());
                    }
                    else {
                        gtChecked = true;
                    }
                    
                }
                
                /*
                 * 2. Check if result region corresponds to > 1 person id in GT
                 */
                cv::Mat overlap;
                predictedMask.copyTo(overlap, person);
                
                labelsOverlap.clear();
                findUniqueValues(overlap, labelsOverlap);
                
                resultRegionMask.release();
                for(int i = 0; i < labelsOverlap.size(); i++)
                {
                    add(resultRegionMask, predictedMask == labelsOverlap[i], resultRegionMask);
                }
                threshold(resultRegionMask,resultRegionMask,128,255,CV_THRESH_BINARY);
                
                cv::Mat gtRegionsOverlap;
                gtMask.copyTo(gtRegionsOverlap,resultRegionMask);
                
                vector<int> gtPersonsOverlap;
                findUniqueValues(gtRegionsOverlap, gtPersonsOverlap);
                gtPersonsOverlap.erase(remove(gtPersonsOverlap.begin(), gtPersonsOverlap.end(), 0), gtPersonsOverlap.end());
                
                if (gtPersonsOverlap.empty() || gtPersonsOverlap.size() == personsInBlob.size())
                {
                    allChecked = true;
                }
                else
                {
                    set<int> s_gtPersonsOverlap (gtPersonsOverlap.begin(), gtPersonsOverlap.end());
                    set<int> s_personsInBlob (personsInBlob.begin(), personsInBlob.end());
                    vector<int> newPersons;
                    
                    set_difference(s_gtPersonsOverlap.begin(), s_gtPersonsOverlap.end(), s_personsInBlob.begin(), s_personsInBlob.end(), back_inserter(newPersons));
                    
                    person.release();
                    for(int i = 0; i < newPersons.size(); i++)
                    {
                        add(person, gtMask == newPersons[i], person);
                    }
                    threshold(person, person, 128, 255, CV_THRESH_BINARY);

                    personsInBlob.insert(personsInBlob.end(), newPersons.begin(), newPersons.end());
                    
                    overlap.release();
                    predictedMask.copyTo(overlap, person);
                    
                }
            }
            
            cv::Mat auxOverlap;
            predictedMask.copyTo(auxOverlap, person);
            if (cv::countNonZero(auxOverlap) > 0)
            {
                std::vector<int>::iterator it;
                for(int i = 0; i < labelsOverlap.size(); i++)
                {
                    it = find(predictedMaskPersonID.begin(), predictedMaskPersonID.end(), labelsOverlap[i]);
                    predictedMaskPersonID.erase(it);
                    
                }
                
                std::vector<int>::iterator it2;
                for(int i = 0; i < personsInBlob.size(); i++)
                {
                    it2 = find(gtMaskPersonID.begin(), gtMaskPersonID.end(), personsInBlob[i]);
                    gtMaskPersonID.erase(it2);
                }
                
                if(cv::countNonZero(dontCareRegion) > 0) //!dontCareRegion.empty()
                {
                    person.copyTo(person, dontCareRegion);
                    resultRegionMask.copyTo(resultRegionMask, dontCareRegion);
                }
                
                //Compute intersection and union
                double intersectArea = 0, unionArea = 0;
                
                cv::Mat intersection, unio;
                cv::bitwise_and(person, resultRegionMask, intersection);
                cv::bitwise_or(person, resultRegionMask, unio);
                
                vector<vector<cv::Point> > intersectContours, unionContours;
                
                findContours(intersection, intersectContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                for(int c = 0; c < intersectContours.size(); c++)
                {
                    intersectArea = intersectArea + contourArea(intersectContours[c]);
                }
                
                findContours(unio, unionContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                for(int c = 0; c < unionContours.size(); c++)
                {
                    unionArea = unionArea + contourArea(unionContours[c]);
                }
                
                if(unionArea > 0) {
                    nBB++;
                    overlapIDs.push_back(intersectArea/unionArea);
                }
            }
            else
            {
                nBB++;
                overlapIDs.push_back(0.0);
                
                std::vector<int>::iterator it2;
                for(int i = 0; i < personsInBlob.size(); i++)
                {
                    it2 = find(gtMaskPersonID.begin(), gtMaskPersonID.end(), personsInBlob[i]);
                    gtMaskPersonID.erase(it2);
                }
            }
        }
        
        int nUnassignedRegions = predictedMaskPersonID.size();
        
        vector<int> unassignedRegions (nUnassignedRegions);
        std::fill(unassignedRegions.begin(), unassignedRegions.end(), 0.0);
        
        overlapIDs.insert(overlapIDs.end(), unassignedRegions.begin(), unassignedRegions.end());
        
        nBB = nBB + nUnassignedRegions;
        
        overlap = std::accumulate(overlapIDs.begin(), overlapIDs.end(), 0)/nBB;
    }
    
    return overlap;
}

void Validation::createDontCareRegion(cv::Mat inputMask, cv::Mat& outputMask, float size)
{
    cv::Mat mask1 (inputMask);
    cv::Mat mask2 (inputMask);
    
    if(size < 1) {
        size = 1;
    }
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*size, 2*size));
    
    cv::morphologyEx(mask1, mask1, cv::MORPH_DILATE, element);
    cv::morphologyEx(mask2, mask2, cv::MORPH_ERODE, element);
    
    subtract(mask1, mask2, outputMask);
    
    threshold(outputMask, outputMask, 128, 255, CV_THRESH_BINARY_INV);
    
    //debug
    cv::imshow("dontCareRegion", outputMask);
    cv::waitKey(10);

}

