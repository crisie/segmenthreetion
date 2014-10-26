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

Validation::Validation(vector<int> dcRange)
: m_DontCareRange(dcRange)
{   }

void Validation::setDontCareRange(vector<int> dcRange)
{
    m_DontCareRange = dcRange;
}

vector<int>& Validation::getDontCareRange()
{
    return m_DontCareRange;
}


void Validation::getOverlap(ModalityData& md, cv::Mat& overlapIDs)
{
    cv::Mat newOverlapIDs(cvSize(m_DontCareRange.size()+1, md.getPredictedMasks().size()), CV_32FC1, NAN);
    this->getOverlap(md.getPredictedMasks(), md.getGroundTruthMasks(), m_DontCareRange, newOverlapIDs);
    if(overlapIDs.empty())
    {
        newOverlapIDs.copyTo(overlapIDs);
    } else
    {
        cv::vconcat(overlapIDs, newOverlapIDs, overlapIDs);
    }
}

void Validation::getOverlap(vector<cv::Mat>& predictedMasks, vector<cv::Mat>& gtMasks, vector<int>& dcRange, cv::Mat& overlapIDs) {
    
    for(int f = 0; f < predictedMasks.size(); f++)
    {
        {
            //debug
            /*vector<int> un;
             findUniqueValues(predictedMasks[f], un);
             for(int a = 0; a < un.size(); a++)
             {
             cout << un[a] << " ";
             }
             cout << endl;*/
            if((cv::countNonZero(gtMasks[f]) == 0 && cv::countNonZero(predictedMasks[f]) > 0)
               || cv::countNonZero(gtMasks[f]) > 0)
            {
                cv::Mat emptyDontCare;
                overlapIDs.at<float>(f,0) = getMaskOverlap(predictedMasks[f], gtMasks[f], emptyDontCare);
                
                for(int dc = 0; dc < dcRange.size(); dc++)
                {
                    cv::Mat dontCare;
                    createDontCareRegion(gtMasks[f], dontCare, dcRange[dc]);
                    //threshold(dontCare, dontCare, 128, 255, CV_THRESH_BINARY);
                    
                    overlapIDs.at<float>(f,dc+1) = getMaskOverlap(predictedMasks[f], gtMasks[f], dontCare);
                    dontCare.release();
                }
            }
            //debug
            /*cout << "overlap f: " << f << " : ";
            for(int a = 0; a < dcRange.size()+1; a++)
            {
                cout << std::setprecision(6) << overlapIDs.at<float>(f,a) << " ";
            }
            cout << endl;*/
            
        }
    }
    
}

void Validation::save(vector<cv::Mat> overlapIDs, cv::Mat meanOverlap, string filename)
{
    vector<cv::Mat> overlapConcat;
    
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    
	for (int fold = 0; fold < overlapIDs.size(); fold++)
	{
        std::stringstream f;
        f << "fold" << fold;
        fs << f.str() <<  overlapIDs[fold];
	}
    
    fs << "meanOverlap" << meanOverlap;
    
	fs.release();
    
}

void Validation::createOverlapPartitions(cv::Mat& partitions, cv::Mat& overlapIDs, vector<cv::Mat>& partitionedOverlapIDs)
{
    partitionedOverlapIDs.clear();
    
    cout << "partitions: " << partitions.cols << " " << partitions.rows << " " << partitions.size() << endl;
    vector<int> folds;
    findUniqueValues(partitions, folds);
    
    for(int f = 0; f < folds.size(); f++)
    {
        cout << "fold " << f << endl;
        cv::Mat matFold;
        
        for(int m = 0; m < partitions.rows; m++)
        {
            if(partitions.at<int>(m,0) == f)
            {
                //cout << partitions.at<int>(m,0) << " ";
                matFold.push_back(overlapIDs.row(m));
            }
            
        }
        //cout << endl;
        partitionedOverlapIDs.push_back(matFold);
    }
}


void Validation::getMeanOverlap(vector<cv::Mat> partitionedOverlapIDs, cv::Mat& partitionedMeanOverlap)
{
    partitionedMeanOverlap.release();
    
    //cv::reduce(overlapIDs, meanOverlap, 1, CV_REDUCE_AVG, -1);
    //std::accumulate(overlapIDs.begin(), overlapIDs.end(), 0.0)/nBB;
    //meanOverlap(cvSize(1, dontCareRange.size()+1), CV_32FC1);
    
    partitionedMeanOverlap = cv::Mat(cvSize(m_DontCareRange.size()+1, partitionedOverlapIDs.size()), CV_32FC1);
    
    for (int p = 0; p < partitionedOverlapIDs.size(); p++)
    {
        //cv::Mat meanOverlap(cvSize(m_DontCareRange.size()+1, 1), CV_32FC1);
        
        for(int c = 0; c < partitionedOverlapIDs[p].cols; c++)
        {
            float accRow = 0.0;
            int validNum = 0;
            for(int r = 0; r < partitionedOverlapIDs[p].rows; r++)
            {
                if (cv::checkRange(partitionedOverlapIDs[p].at<float>(r,c)))
                {
                    accRow += partitionedOverlapIDs[p].at<float>(r,c);
                    validNum++;
                }
            }
            partitionedMeanOverlap.at<float>(p,c) = accRow/validNum;
        }
        
        //debug
        cout << "Mean overlap partition " << p << ":";
        for(int i = 0; i < partitionedMeanOverlap.cols; i++)
        {
            cout << std::setprecision(6) << partitionedMeanOverlap.at<float>(p,i) << " ";
        }
        cout << endl;
    }
    
}

/*
 * Get overlap value between predicted mask and ground truth mask based on Jaccard Similarity/Index
 */
float Validation::getMaskOverlap(cv::Mat& predictedMask, cv::Mat& gtMask, cv::Mat& dontCareRegion)
{
    
    // imshow("gtMask", gtMask);
    // imshow("predictedMask", predictedMask);
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
    
    if(!predictedMaskPersonID.empty())
    {
        std::vector<int>::iterator gtID = gtMaskPersonID.begin();
        while (gtID != gtMaskPersonID.end())
        {
            int id  = *gtID;
            cv::Mat person = (gtMask == id);
            //threshold(person, person, 128, 255, CV_THRESH_BINARY);
            //imshow("person", person);
            
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
                        //imshow("person", person);
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
                //imshow("resultRegionMask", resultRegionMask);
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
                    
                    //person.release();
                    //person = cv::Mat::zeros(gtMask.rows, gtMask.cols, CV_8UC1);
                    for(int i = 0; i < newPersons.size(); i++)
                    {
                        add(person, gtMask == newPersons[i], person);
                    }
                    //threshold(person, person, 128, 255, CV_THRESH_BINARY);
                    //imshow("person2", person);
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
                //cv::imshow("intersection", intersection);
                //cv::imshow("union", unio);
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
    
    /*if(size < 1) {
     size = 1;
     }*/
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*size+1, 2*size+1));
    
    cv::morphologyEx(mask1, mask1, cv::MORPH_DILATE, element);
    
    cv::morphologyEx(mask2, mask2, cv::MORPH_ERODE, element);
    
    subtract(mask1, mask2, outputMask);
    
    threshold(outputMask, outputMask, 128, 255, CV_THRESH_BINARY_INV);
    
    mask1.release();
    mask2.release();
    element.release();
    
    
}

