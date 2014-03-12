//
//  DepthBackgroundSubtractor.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 05/03/14.
//
//

#include "BackgroundSubtractor.h"
#include "DepthBackgroundSubtractor.h"
#include "StatTools.h"

#include <opencv2/opencv.hpp>

DepthBackgroundSubtractor::DepthBackgroundSubtractor()
: BackgroundSubtractor()
{ }

/*
 DepthBackgroundSubtractor::DepthBackgroundSubtractor(vector<int> numFramesToLearn, unsigned char masksOffset)
 : BackgroundSubtractor(), m_numFramesToLearn(numFramesToLearn), m_masksOffset(masksOffset)
 { }
 */

DepthBackgroundSubtractor::DepthBackgroundSubtractor(ForegroundParametrization fParam)
: BackgroundSubtractor(), m_fParam(fParam)
{ }

/*
 void DepthBackgroundSubtractor::setNumFramesToLearn(vector<int> numFramesToLearn) {
 
 m_numFramesToLearn = numFramesToLearn;
 }
 */

/*void DepthBackgroundSubtractor::setMasksOffset(unsigned char masksOffset)
{
    m_masksOffset = masksOffset;
}*/

void DepthBackgroundSubtractor::getMasks(ModalityData& md) {
    
    vector<cv::Mat> masks;
    
    //Per scene
    for(int scene = 0; scene < md.getNumScenes(); scene++) {
        
        //Initialize BackgroundSubtractorMOG2 class
        int history = m_fParam.numFramesToLearn[scene];
        float varThreshold = 3;
        cv::BackgroundSubtractorMOG2 bgsubtractor(history, varThreshold, false);
        
        //Initialize variables for BS loop
        cv::Mat output, outputContours;
        
        //Per frame in scene
        for(int f = 0; f < md.getFramesInScene(scene).size(); f++) {
            
            vector<cv::Mat> outputMasksTemp;
            
            cv::Mat frame (md.getFrameInScene(scene, f));
            
            if(f < m_fParam.numFramesToLearn[scene])
            {
                bgsubtractor(frame, output, f == 0 ? 1 : 0.02);
                output = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
                outputMasksTemp.push_back(cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1));
            }
            else
            {
                output = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
                bgsubtractor(frame, output, 0);
                
                this->extractItemsFromMask(frame, output);
                
                //TODO: visualization purposes
                cv::Mat auxOutput;
                frame.copyTo(auxOutput);
                
                //...
                
            }
            masks.push_back(output);
        }
    }
    
    md.setPredictedMasks(masks);
    //md.setMasks(masks);
    
}

void DepthBackgroundSubtractor::getBoundingRects(ModalityData& md) {
    
    vector<vector<cv::Rect> > boundingRects(md.getFrames().size());
    
    for(unsigned int f = 0; f < md.getFrames().size(); f++) {
        
        cv::Mat mask = md.getMask(f);
        
        vector<int> uniqueValuesMask;
        findUniqueValues(mask, uniqueValuesMask);
        
        for(unsigned int i = 0; i < uniqueValuesMask.size(); i++)
        {
            cv::Rect bigBoundingBox;
            vector<cv::Rect> maskBoundingBoxes;
            this->getMaskBoundingBoxes(mask, maskBoundingBoxes);
            
            if(!maskBoundingBoxes.empty()) {
                this->getMaximalBoundingBox(maskBoundingBoxes, bigBoundingBox);
                
                if(!this->checkMinimumBoundingBoxes(bigBoundingBox, 4)) {
                    boundingRects[f].push_back(getMinimumBoundingBox(bigBoundingBox, 4));
                } else {
                    boundingRects[f].push_back(bigBoundingBox);
                }
            }
            
        }
        
    }
    
    md.setBoundingRects(boundingRects);
    
    cout << "Depth bounding boxes: " << this->countBoundingBoxes(md.getBoundingRects()) << endl;

    
}

void DepthBackgroundSubtractor::extractItemsFromMask(cv::Mat frame, cv::Mat & mask){
    
    cv::Mat valuedMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    vector<pair<cv::Mat, cv::Rect> > item;
    unsigned int nItem = 0;
    
    
    float bbMinArea = (frame.rows*frame.cols) * m_fParam.boundingBoxMinArea;
	float otsuMinArea = (frame.rows*frame.cols) * m_fParam.otsuMinArea;
    
    float interiorHolesMinArea = 100.0; //200
    
    vector<vector<cv::Point> > contours;
	vector<vector<cv::Point> > filteredContours;
	vector<vector<cv::Point> > otsuContours;
	vector<vector<cv::Point> > auxOtsuContours;
    
	cv::Mat kernel3x3 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::Mat kernel9x9 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    
    //Opening
    cv::morphologyEx(mask,mask, cv::MORPH_ERODE, kernel3x3);
    cv::morphologyEx(mask,mask, cv::MORPH_DILATE, kernel3x3);
    
    cv::Mat mask2;
    mask.copyTo(mask2);
	vector<cv::Vec4i> hierarchy;
	findContours(mask2, //binary output image
                 contours, //vector of vectors of points
                 hierarchy, //hierarchy
                 CV_RETR_CCOMP, //CV_RETR_EXTERNAL retrieve only external contours
                 CV_CHAIN_APPROX_NONE); //detect all pixels of each contour
    
    
    
    //Detect little interior holes
    cv::Mat singleLevelHoles = cv::Mat::zeros(mask2.size(), mask2.type());
	vector<vector<cv::Point> > contours_poly(contours.size());
	vector<cv::Rect> boundRect2(contours.size());
	for(vector<cv::Vec4i>::size_type idx=0; idx<hierarchy.size(); ++idx)
	{
		if(hierarchy[idx][3] != -1) {
            approxPolyDP( cv::Mat(contours[idx]), contours_poly[idx], 2, true );
            boundRect2[idx] = boundingRect( cv::Mat(contours_poly[idx]) );
            if(fabs(contourArea(cv::Mat(contours_poly[idx]))) > interiorHolesMinArea) {
                drawContours(singleLevelHoles, contours,idx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy);
                morphologyEx(singleLevelHoles,singleLevelHoles, cv::MORPH_OPEN, kernel3x3);
                
            }
		}
	}
    
    contours.clear();
    contours_poly.clear();
    
    findContours(mask2, //binary output image
                 contours, //vector of vectors of points
                 CV_RETR_EXTERNAL, //CV_RETR_EXTERNAL retrieve only external contours
                 CV_CHAIN_APPROX_NONE); //detect all pixels of each contour
    
    contours_poly.resize(contours.size());
	vector<cv::Rect> boundRect(contours.size());
    
    //Draw polygonal contours
	for( unsigned int i = 0; i< contours.size(); i++ )
	{
        approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 2, true );
        
		boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
		double blobArea = fabs(contourArea(cv::Mat(contours_poly[i])));
        
        //filter very tiny blobs which are not in the sides of the image - they may be incoming objects or people
		if(blobArea > bbMinArea ||
           ((boundRect[i].tl().x < 55 || boundRect[i].br().x > frame.cols - 15) && blobArea > bbMinArea/2))
		{
            cv::Mat roiMask, frameGray, otsuMask;
			cvtColor(frame,frameGray,CV_RGB2GRAY);
			roiMask = cv::Mat::zeros(frame.size(),CV_8UC1);
            
			drawContours( roiMask, contours, i, cv::Scalar::all(255), CV_FILLED, 8, vector<cv::Vec4i>());
            
            //Subtract little holes
            cv::subtract(roiMask, singleLevelHoles, roiMask);
			morphologyEx(roiMask,roiMask, cv::MORPH_CLOSE, kernel3x3);
            
            //Treat white regions (zero values) before Otsu algorithm
            cv::Mat roi, contourRegion, whiteRegion, blackRegion;
			frameGray.copyTo(roi,roiMask);
			threshold(roi,blackRegion,200,255,CV_THRESH_TOZERO_INV);
			threshold(roi,whiteRegion,200,255,CV_THRESH_BINARY);
            
            cv::Mat blackRegionMask;
			threshold(blackRegion, blackRegionMask, 1, 255, CV_THRESH_BINARY);
            
            cv::Scalar roi_mean = mean(blackRegion, blackRegionMask);
            roi.setTo(roi_mean.val[0], whiteRegion);
            
            //Mask out zero values before computing the optimal threshold with Otsu algorithm
            cv::Mat roiCopy;
			roi.copyTo(roiCopy);
			uchar * ptr = roiCopy.datastart;
			uchar * ptr_end = roiCopy.dataend;
			while (ptr < ptr_end) {
				if (*ptr == 0) { // swap if zero
					unsigned char tmp = *ptr_end;
					*ptr_end = *ptr;
					*ptr = tmp;
					ptr_end--; // make array smaller
				} else {
					ptr++;
				}
			}
            
            //Calculate mean and std using non-zero values
			//New matrix with only valid data
            cv::Scalar mean, std;
            cv::Mat nonZeroRoi = cv::Mat(vector<unsigned char>(roiCopy.datastart,ptr_end),true);
			meanStdDev(nonZeroRoi, mean, std);
            
            
            contours_poly.clear();
            cv::Mat auxOtsuMask;
            
            //If there is a region with std larger than otsuMinVariance, use OTSU threshold to
			//divide the region
            if((std.val[0] > m_fParam.otsuMinVariance1 && boundRect[i].area() > otsuMinArea )
               || std.val[0] > m_fParam.otsuMinVariance2) {
                
                threshold(roi,auxOtsuMask,1,255.0,CV_THRESH_BINARY);
                
                //find Otsu threshold and apply it to divide the region
                double thresh = threshold(nonZeroRoi,nonZeroRoi,-1,255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
				threshold(roi,otsuMask,thresh,255.0,CV_THRESH_TOZERO_INV);
				threshold(otsuMask,otsuMask,1,255.0,CV_THRESH_BINARY);
                
                cv::subtract(otsuMask, whiteRegion, otsuMask);
				morphologyEx(otsuMask, otsuMask, cv::MORPH_OPEN, kernel9x9);
                cv::subtract(auxOtsuMask, otsuMask, auxOtsuMask);
                
                cv::subtract(auxOtsuMask, whiteRegion, auxOtsuMask);
				morphologyEx(auxOtsuMask, auxOtsuMask, cv::MORPH_OPEN, kernel9x9);
                
                //Find largest component that overlaps
                cv::Mat auxOtsuMask2;
                auxOtsuMask.copyTo(auxOtsuMask2);
                
				findContours(auxOtsuMask2, //binary output image
                             auxOtsuContours, //vector of vectors of points
                             CV_RETR_EXTERNAL, //retrieve only external contours
                             CV_CHAIN_APPROX_NONE); //detect all pixels of each contour
                
                
                vector<vector<cv::Point> > largestComps;
                cv::Mat bgOtsuLargestComp = cv::Mat::zeros(frame.size(), CV_8UC1);
                for(unsigned int idx = 0; idx < auxOtsuContours.size(); idx++) {
					const vector<cv::Point>& c = auxOtsuContours[idx];
					double area = fabs(contourArea(cv::Mat(c)));
					if (area > 0.003 * frame.cols * frame.rows) {
						largestComps.push_back(c);
						drawContours(bgOtsuLargestComp, auxOtsuContours, idx, 255, CV_FILLED, 8, vector<cv::Vec4i>());
					}
				}
                
                //One big minimal bounding box: store the set of points in the image before assembling the bounding box
                vector<cv::Point> points;
                cv::Mat_<uchar>::iterator it_start = bgOtsuLargestComp.begin<uchar>();
                cv::Mat_<uchar>::iterator it_end = bgOtsuLargestComp.end<uchar>();
				for(; it_start != it_end; ++it_start) {
					if (*it_start == 255) {
                        points.push_back(it_start.pos());
                    }
				}
                
                //Compute minimal bounding box
                cv::Rect tempRoiBox;
				if(!points.empty()) {
					tempRoiBox = boundingRect(cv::Mat(points));
                    if(tempRoiBox.area() > 0.4 * frame.cols * frame.rows) {
                        contours_poly.resize(largestComps.size());
                        for(unsigned int idxComp = 0; idxComp < largestComps.size(); idxComp ++) {
                            approxPolyDP( cv::Mat(largestComps[idxComp]), contours_poly[idxComp], 2, true );
                            
                            cv::Mat tempMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                            drawContours(tempMask, largestComps, idxComp, 255, CV_FILLED, 8, vector<cv::Vec4i>());
                            item.push_back(std::make_pair(tempMask, boundingRect( cv::Mat(contours_poly[idxComp]) )));
                        }
                    }
                    else {
                        item.push_back(std::make_pair(bgOtsuLargestComp,  tempRoiBox));
                        
                    }
				}
                
                auxOtsuContours.clear();
                points.clear();
                
                //Find other components
                
                cv::Mat otsuMask2;
				otsuMask.copyTo(otsuMask2);
                findContours(otsuMask2, //binary output image
                             auxOtsuContours, //vector of vectors of points
                             CV_RETR_EXTERNAL, //retrieve only external contours
                             CV_CHAIN_APPROX_NONE); //detect all pixels of each contour
                
                cv::Mat otsuLargestComp = cv::Mat::zeros(frame.size(), CV_8UC1);
				for(unsigned int idx = 0; idx < auxOtsuContours.size(); idx++) {
					const vector<cv::Point>& c = auxOtsuContours[idx];
					double area = fabs(contourArea(cv::Mat(c)));
					if (area > 0.003 * frame.cols * frame.rows) {
						drawContours(otsuLargestComp, auxOtsuContours, idx, 255, CV_FILLED, 8, vector<cv::Vec4i>());
                        
					}
				}
                
				cv::Mat_<uchar>::iterator it_start2 = otsuLargestComp.begin<uchar>();
				cv::Mat_<uchar>::iterator it_end2 = otsuLargestComp.end<uchar>();
				for(; it_start2 != it_end2; ++it_start2) {
					if (*it_start2 == 255) {
                        points.push_back(it_start2.pos());
                    }
				}
                
                if(!points.empty()) {
                    item.push_back(std::make_pair(otsuLargestComp,  boundingRect(cv::Mat(points))));
				}
                
            }
            //If the region has not a large std, obtain bounding boxes as usual
            else
            {
                threshold(roiMask,otsuMask,1,255,CV_THRESH_BINARY);
                findContours(otsuMask, otsuContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                
				for(unsigned int j = 0; j < otsuContours.size(); j++)
				{
                    cv::Mat tempMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    
                    drawContours(tempMask, otsuContours, j, 255, CV_FILLED, 8, vector<cv::Vec4i>());
                    cv::subtract(tempMask, singleLevelHoles, tempMask);
					approxPolyDP( cv::Mat(otsuContours[j]), contours_poly, 2, true );
                    item.push_back(std::make_pair(tempMask, boundingRect( cv::Mat(contours_poly)) ));
                    
				}
            }
            
        }
        
    }
    
    for(unsigned int o = 0; o < item.size(); o++) {
        if(item[o].second.area() > 9 && item[o].second.width > 2 && item[o].second.height > 2) {
            this->changePixelValue(item[o].first, nItem);
            this->checkWhitePixels(item[o].second, frame);
            
            //masksCollection.push_back(item[o].first);
            //boundingBoxes.push_back(item[o].second);
            
            add(valuedMask, item[o].first, valuedMask);
            
            //Debug purposes - show mask unique valuess
            vector<int> uniqueValues;
            findUniqueValues(item[o].first, uniqueValues);
            for (auto c : uniqueValues)
                cout << c << ' ';
            
            nItem++;
        }
    }
    
    valuedMask.copyTo(mask);
    
    //Debug purposes - show mask unique values
    vector<int> uniqueValues;
    findUniqueValues(mask, uniqueValues);
    for (auto c : uniqueValues)
        cout << c << ' ';
    
}

void DepthBackgroundSubtractor::adaptGroundTruthToReg(ModalityData& md) {
    
    vector<cv::Mat> newDepthGtMasks;
    
    for(int s = 0; s < md.getNumScenes(); s++)
    {
        vector<cv::Mat> depthGtMasks = md.getGroundTruthMasksInScene(s);
        vector<cv::Mat> depthFrames = md.getFramesInScene(s);
        
        cv::Mat mask;
        
        threshold(md.getFrameInScene(s,0),mask,1,255,CV_THRESH_BINARY);
        
        for(unsigned int f = 0; f < md.getGroundTruthMasksInScene(s).size(); f++) {
            
            cv::Mat gtMask = md.getGroundTruthMaskInScene(s,f);
            
            cv::Mat frameAux = cv::Mat::zeros(gtMask.size(), gtMask.type());
            depthGtMasks[f].copyTo(frameAux, mask);
            
            newDepthGtMasks.push_back(frameAux);
            
        }

    }
    
    md.setGroundTruthMasks(newDepthGtMasks);
    
    //TODO: save masks somehow (?)
}


