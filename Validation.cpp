//
//  Validation.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 21/03/14.
//
//

#include "Validation.h"
#include "StatTools.h"
#include "CvExtraTools.h"

#include <iomanip>


using namespace std;

Validation::Validation()
{ }

void Validation::getOverlap(ModalityData& md, vector<int>& dcRange, cv::Mat& overlapIDs) {
    
    cv::Mat newOverlapIDs(cvSize(md.getPredictedMasks().size(), dcRange.size()+1), CV_32FC1, NAN);
    this->getOverlap(md.getPredictedMasks(), md.getGroundTruthMasks(), dcRange, newOverlapIDs);
    cv::hconcat(overlapIDs, newOverlapIDs, overlapIDs);
    
}

void Validation::getOverlap(vector<cv::Mat>& predictedMasks, vector<cv::Mat>& gtMasks, vector<int>& dcRange, cv::Mat& overlapIDs) {
    
    //cv::resize(overlapIDs, overlapIDs, cvSize(predictedMasks.size(), dcRange.size()+1));
    int idx = 0;
    
    for(int f = 0; f < predictedMasks.size(); f++)
    {
        //debug
        //cout << "predicted mask " << predictedMasks[f].channels() << " " << predictedMasks[f].type() << endl;
        //cout << "gt mask " << gtMasks[f].channels() << " " << gtMasks[f].type() << endl;
        if(cv::countNonZero(predictedMasks[f]) > 0 || cv::countNonZero(gtMasks[f]) > 0)
        {
            //debug
            vector<int> un;
            findUniqueValues(predictedMasks[f], un);
            for(int a = 0; a < un.size(); a++)
            {
                cout << un[a] << " ";
            }
            cout << endl;
            
            cv::Mat emptyDontCare;
            overlapIDs.at<float>(0, idx) = getMaskOverlap(predictedMasks[f], gtMasks[f], emptyDontCare);
            
            for(int dc = 0; dc < dcRange.size(); dc++)
            {
                cv::Mat dontCare;
                createDontCareRegion(gtMasks[f], dontCare, dcRange[dc]);
                //threshold(dontCare, dontCare, 128, 255, CV_THRESH_BINARY);
                
                overlapIDs.at<float>(dc+1, idx) = getMaskOverlap(predictedMasks[f], gtMasks[f], dontCare);
                dontCare.release();
            }
            
            cout << "overlap f: " << f << " : ";
            for(int a = 0; a < dcRange.size()+1; a++)
            {
                cout << std::setprecision(6) << overlapIDs.at<float>(a,idx) << " ";
            }
            cout << endl;
            
            idx++;
        }
    }
    
}

void Validation::save(cv::Mat overlapIDs, cv::Mat meanOverlap, string filename)
{
    cv::Mat overlapConcat;
    cv::vconcat(overlapIDs, meanOverlap, overlapConcat);
    
    cvx::save(filename, overlapConcat);
}


void Validation::getMeanOverlap(cv::Mat overlapIDs, cv::Mat& meanOverlap)
{
    //cv::reduce(overlapIDs, meanOverlap, 1, CV_REDUCE_AVG, -1);
    //std::accumulate(overlapIDs.begin(), overlapIDs.end(), 0.0)/nBB;
    for(int r = 0; r < overlapIDs.rows; r++)
    {
        float accRow = 0.0;
        int idx = 0;
        for(int f = 0; f < overlapIDs.cols; f++)
        {
            if (cv::checkRange(overlapIDs.at<float>(r,f)))
            {
                accRow += overlapIDs.at<float>(r,f);
                idx++;
            }
        }
        meanOverlap.at<float>(r,0) = accRow/idx;
    }
}

/*
 * Get overlap value between predicted mask and ground truth mask based on Jaccard Similarity/Index
 */
float Validation::getMaskOverlap(cv::Mat& predictedMask, cv::Mat& gtMask, cv::Mat& dontCareRegion)
{
    
  //  imshow("gtMask", gtMask);
  //  imshow("predictedMask", predictedMask);
    cv::waitKey(10);
    
    bool useDontCare = false;
    if(!dontCareRegion.empty()) useDontCare = true;
    
    float overlap = 0.0;
    int nBB = 0;
    vector<float> overlapIDs;
    
    vector<int> gtMaskPersonID;
    findUniqueValues(gtMask, gtMaskPersonID);
    gtMaskPersonID.erase(std::remove(gtMaskPersonID.begin(), gtMaskPersonID.end(), 0), gtMaskPersonID.end());
    
    //labeled_result_mask: we already have one label per blob (200, 201, 202....)
    vector<int> predictedMaskPersonID;
    findUniqueValues(predictedMask, predictedMaskPersonID);
    predictedMaskPersonID.erase(std::remove(predictedMaskPersonID.begin(), predictedMaskPersonID.end(), 0), predictedMaskPersonID.end());
        
   /* cv::Mat labeledGtMask = cv::Mat::zeros(gtMask.size(), CV_8UC1); //TOCHECK: gtMask is already labeled?
    vector<vector<cv::Point> > contours;
    cv::Mat auxGtMask;
    gtMask.copyTo(auxGtMask);
    imshow("auxGtMask", auxGtMask);
    findContours(auxGtMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    for(int c = 0; c < contours.size(); c++)
    {
        drawContours(labeledGtMask, contours, c, c, CV_FILLED, 8, vector<cv::Vec4i>());
    }
    imshow("labeledGtMask", labeledGtMask);*/
    //if(cv::waitKey() > 30) {}
    vector<bool> gtMaskUsedIDs;
    for(int g = 0; g < gtMaskPersonID.size(); g++) gtMaskUsedIDs.push_back(false);
    
    vector<bool> prMaskUsedIDs;
    for(int g = 0; g < predictedMaskPersonID.size(); g++) prMaskUsedIDs.push_back(false);
    
    if(!predictedMaskPersonID.empty())
    {
        std::vector<int>::iterator gtID = gtMaskPersonID.begin();
        while (gtID != gtMaskPersonID.end())
        {
            int id  = *gtID;
            cv::Mat person = (gtMask == id);
            //threshold(person, person, 128, 255, CV_THRESH_BINARY);
        //    imshow("person", person);
            
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
                    gtMask.copyTo(maskedLabeledGtMask, person);
                    findUniqueValues(maskedLabeledGtMask, regionIDs);
                    regionIDs.erase(std::remove(regionIDs.begin(), regionIDs.end(), 0), regionIDs.end());
                    
                    for(int i = 0 ; i < regionIDs.size(); i++)
                    {
                        vector<int> uniqueIDs;
                        cv::Mat gtMaskContainingID = (gtMask == regionIDs[i]);

                        for(int r = 0; r < gtMaskContainingID.rows; r++) for(int c = 0; c < gtMaskContainingID.cols; c++)
                        {
                            if(gtMaskContainingID.at<uchar>(r,c) == 255)
                                gtMaskContainingID.at<uchar>(r,c) = regionIDs[i];
                        }
                        
                        findUniqueValues(gtMaskContainingID, uniqueIDs);
                        uniqueIDs.erase(std::remove(uniqueIDs.begin(), uniqueIDs.end(), 0), uniqueIDs.end());
                        personsInBlob.insert(personsInBlob.end(), uniqueIDs.begin(), uniqueIDs.end());
                    }
                    
                    findUniqueValues(personsInBlob, personsInBlob);
                    
                    if(personsGt.size() != personsInBlob.size())
                    {
                        person.release();
                        person = cv::Mat::zeros(gtMask.rows, gtMask.cols, CV_8UC1);
                        for(int i = 0; i < personsInBlob.size(); i++)
                        {
                            add(person, gtMask == personsInBlob[i], person);
                        }
                        //threshold(person, person, 128, 255, CV_THRESH_BINARY);
                    //    imshow("person", person);
                        cv::waitKey(10);
                        
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
                labelsOverlap.erase(remove(labelsOverlap.begin(), labelsOverlap.end(), 0), labelsOverlap.end());
                
                resultRegionMask.release();
                resultRegionMask = cv::Mat::zeros(gtMask.rows, gtMask.cols, CV_8UC1);
                for(int i = 0; i < labelsOverlap.size(); i++)
                {
                    add(resultRegionMask, predictedMask == labelsOverlap[i], resultRegionMask);
                }
                //threshold(resultRegionMask,resultRegionMask,128,255,CV_THRESH_BINARY);
               // imshow("resultRegionMask", resultRegionMask);
                cv::waitKey(10);
                
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
                    person = cv::Mat::zeros(gtMask.rows, gtMask.cols, CV_8UC1);
                    for(int i = 0; i < newPersons.size(); i++)
                    {
                        add(person, gtMask == newPersons[i], person);
                    }
                    //threshold(person, person, 128, 255, CV_THRESH_BINARY);
                  //  imshow("person2", person);
                    cv::waitKey(10);
                    
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
                
                if(useDontCare)
                {
                    person = person & dontCareRegion;
                    resultRegionMask = resultRegionMask & dontCareRegion;

                }
                //debug
             //   cv::imshow("person after dontcare", person);
             //   cv::imshow("resultRegionMask after dontcare", resultRegionMask);
                cv::waitKey(10);
                
                //Compute intersection and union
                float intersectArea = 0.0, unionArea = 0.0;
                
                cv::Mat intersection, unio;
                cv::bitwise_and(person, resultRegionMask, intersection);
                cv::bitwise_or(person, resultRegionMask, unio);
                
                //debug
             //   cv::imshow("intersection", intersection);
             //   cv::imshow("union", unio);
                cv::waitKey(10);
                
                intersectArea = cv::countNonZero(intersection);
                unionArea = cv::countNonZero(unio);
                
                if(unionArea > 0) {
                    nBB++;
                    overlapIDs.push_back(float(intersectArea/unionArea));
                }

                person.release();
                resultRegionMask.release();
                intersection.release();
                unio.release();
                auxOverlap.release();
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
            
            cv::destroyAllWindows();
        }
        
        int nUnassignedRegions = predictedMaskPersonID.size();
        
        if(nUnassignedRegions > 0)
        {
            vector<int> unassignedRegions (nUnassignedRegions);
            std::fill(unassignedRegions.begin(), unassignedRegions.end(), 0.0);
        
            overlapIDs.insert(overlapIDs.end(), unassignedRegions.begin(), unassignedRegions.end());
        
            nBB = nBB + nUnassignedRegions;
            
        }
        
        if(nBB > 0) {
            overlap = std::accumulate(overlapIDs.begin(), overlapIDs.end(), 0.0)/nBB;
        } else {
            overlap = NAN;
        }
    }
   
    cv::destroyAllWindows();
    overlapIDs.clear();
    gtMaskPersonID.clear();
    predictedMaskPersonID.clear();
    
    return overlap;
   }

void Validation::createDontCareRegion(cv::Mat& inputMask, cv::Mat& outputMask, int size)
{
    cv::Mat mask1, mask2;
    inputMask.copyTo(mask1);
    inputMask.copyTo(mask2);
    
    if(size < 1) {
        size = 1;
    }
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*size+1, 2*size+1));
    
    cv::morphologyEx(mask1, mask1, cv::MORPH_DILATE, element);
    
    cv::morphologyEx(mask2, mask2, cv::MORPH_ERODE, element);
    
    subtract(mask1, mask2, outputMask);
    
    threshold(outputMask, outputMask, 128, 255, CV_THRESH_BINARY_INV);
    
    mask1.release();
    mask2.release();
    element.release();


}

