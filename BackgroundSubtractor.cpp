//
//  BackgroundSubtractor.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 05/03/14.
//
//

#include "BackgroundSubtractor.h"


BackgroundSubtractor::BackgroundSubtractor()
{   }

void BackgroundSubtractor::setMasksOffset(unsigned char masksOffset) {
    m_masksOffset = masksOffset;
}

unsigned char BackgroundSubtractor::getMasksOffset() {
    return m_masksOffset;
}

void BackgroundSubtractor::getGroundTruthBoundingRects(ModalityData& md) {
    
    vector<vector<cv::Rect> >  bbModal;
    
    for(int f = 0; f < md.getGroundTruthMasks().size(); f++) {
    
        cv::Mat mask (md.getGroundTruthMask(f));
        vector<cv::Rect> bbTemp;
        
        cvtColor(mask, mask, CV_RGB2GRAY);
        
        
        cv::Mat binaryMask = cv::Mat::zeros(mask.size(), CV_8UC1);
		threshold(mask,binaryMask,128,255,CV_THRESH_BINARY);
        
        vector<vector<cv::Point> > contours;
		findContours(binaryMask,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //find contours in groundtruth
		vector<vector<cv::Point> > contours_poly(contours.size());
		vector<cv::Rect> boundRect(contours.size());
		
        for(unsigned int i = 0; i< contours.size(); i++ )
		{
			//find bounding boxes around ground truth contours
			approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 2, true );
			boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
			bbTemp.push_back(boundRect[i]);
		}
        
        bbModal.push_back(bbTemp);
		bbTemp.clear();
    }
    
    md.setGroundTruthBoundingRects(bbModal);
    
}

void BackgroundSubtractor::getRoiTags(ModalityData& md, bool manualAid) {
    
    vector<vector<int> > bbTags(md.getPredictedMasks().size());
    
    for(unsigned int f = 0; f < md.getPredictedMasks().size(); f++) {
        
        vector<cv::Rect> boundRects = md.getPredictedBoundingRectsInFrame(f);
        vector<cv::Rect> gtBoundRects = md.getGroundTruthBoundingRectsInFrame(f);
        
        if(!boundRects.empty()) {
            
            for(unsigned int b = 0; b < boundRects.size(); b++)
            {
                int isPerson = FALSE;
                for(unsigned int gt = 0; gt < gtBoundRects.size(); gt++)
                {
                    isPerson = isPersonBox(boundRects[b],gtBoundRects[gt]);
                    if(isPerson == TRUE) break;
                }
                
                if(manualAid && isPerson != UNDEFINED) {
                    //cv::namedWindow("Tag ROI manually.");
                    
                    cv::Mat frame = md.getFrame(f);
                    cv::Mat auxFrame;
                    frame.copyTo(auxFrame);
                    imshow("frame", auxFrame); //frame
                    cv::Mat mask = md.getPredictedMask(f);
                    add(auxFrame, cv::Scalar(100, 100, 0), auxFrame, mask);
                    cv::rectangle(auxFrame, boundRects[b].tl(), boundRects[b].br(), cvScalar(255,0,0));
                    cv::imshow("Tag ROI manually", auxFrame);
                    cv::waitKey(10);
                    bool done = false;
                    do {
                        cout << "Manual tagging... Press 'p' if it the bounding box surrounds a person, 'o' if object and 'u' if undefined." << endl;
                        string input = "";
                        getline(cin, input);
                        boost::algorithm::to_lower(input);
                        
                        if (input.compare("p") == 0) {
                            isPerson = TRUE;
                            done = true;
                        }
                        else if (input.compare("o") == 0) {
                            isPerson = FALSE;
                            done = true;
                        }
                        else if(input.compare("u") == 0) {
                            isPerson = UNDEFINED;
                            done = true;
                        }
                        else cout << "Incorrect option." << endl;
                    } while (!done);
                }
                
                bbTags[f].push_back(isPerson);
                
            }
            
        }
        
    }
    
    md.setTags(bbTags);
    
}

void BackgroundSubtractor::changePixelValue(cv::Mat & mask, int pixelValue) {
    
    int threshold = m_masksOffset;
    int newPixelValue = m_masksOffset + pixelValue;
    changePixelValue(mask, threshold, newPixelValue);
}

void BackgroundSubtractor::changePixelValue(cv::Mat & mask, int threshold, int pixelValue) {
    
    if(threshold != m_masksOffset) {
        pixelValue = m_masksOffset + pixelValue;
    }
    
    vector<cv::Point> points;
    cv::Mat_<uchar>::iterator it_start = mask.begin<uchar>();
    cv::Mat_<uchar>::iterator it_end = mask.end<uchar>();
    for(; it_start != it_end; ++it_start) {
        if (*it_start >= threshold) {
            *it_start = pixelValue;
        }
        else if(*it_start < 0) {
            *it_start = 0;
        }
    }
    
}

void BackgroundSubtractor::checkWhitePixels(cv::Rect & box, cv::Mat frame) {
	
    const int boundary = 10;
	
    cv::Point tl(0,0), br(0,0);
	
    if(box.tl().x - boundary < 0) {
		tl.x = 0;
	} else {
		tl.x = box.tl().x - boundary;
	}
	if(box.tl().y - boundary < 0) {
		tl.y = 0;
	} else {
		tl.y = box.tl().y - boundary;
	}
	if(box.br().x + boundary > frame.cols - 1) {
		br.x = frame.cols - 1;
	} else {
		br.x = box.br().x + boundary;
	}
	if(box.br().y + boundary > frame.rows - 1) {
		br.y = frame.rows - 1;
	} else {
		br.y = box.br().y + boundary;
	}
	cv::Rect auxBox(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
	box = auxBox;
    
}

void BackgroundSubtractor::getMaximalBoundingBox(vector<cv::Rect> boundingBox, cv::Rect & outputBoundingBox) {
	
    cv::Point tl(INFINITY, INFINITY);
    cv::Point br(-1,-1);
	
    for(unsigned int i = 0; i < boundingBox.size(); i++) {
		tl = getTopLeftPoint(boundingBox[i].tl(), tl);
		br = getBottomRightPoint(boundingBox[i].br(), br);
	}
    
	outputBoundingBox = cv::Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    
}

cv::Point BackgroundSubtractor::getTopLeftPoint(cv::Point p1, cv::Point p2) {
 	int x = min(p1.x, p2.x);
 	int y = min(p1.y, p2.y);
    
	return cv::Point(x,y);
}

cv::Point BackgroundSubtractor::getBottomRightPoint(cv::Point p1, cv::Point p2) {
	int x = max(p1.x, p2.x);
	int y = max(p1.y, p2.y);
    
	return cv::Point(x,y);
}

int BackgroundSubtractor::countBoundingBoxes(vector<vector<cv::Rect> > boundingBoxes) {
    
    int nCount = 0;
    
    for(unsigned int f = 0; f < boundingBoxes.size(); f++) {
        nCount = nCount + int(boundingBoxes[f].size());
    }
    return nCount;
}

void BackgroundSubtractor::getMaskBoundingBoxes(cv::Mat mask, vector<cv::Rect> & boundingBoxes) {
    
    cv::Mat auxMask;
    mask.copyTo(auxMask);
    
    vector<vector<cv::Point> > contours;
    findContours(auxMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    vector<vector<cv::Point> > contours_poly(contours.size());
    
    // Draw polygonal contour + bounding rects
    for(unsigned int i = 0; i< contours.size(); i++ )
    {
        approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 5, true );
        boundingBoxes.push_back(boundingRect( cv::Mat(contours_poly[i]) ));
    }
    
}

bool BackgroundSubtractor::checkMinimumBoundingBoxes(cv::Rect box, int min)
{
    if(box.height > min && box.width > min) return true;
    else return false;
}

cv::Rect BackgroundSubtractor::getMinimumBoundingBox(cv::Rect box, int min)
{
    int width = box.width, height = box.height;
    
    if(box.height < min) height = min;
    if(box.width < min) width = min;
    
    return cv::Rect(box.tl().x, box.tl().y, min, min);
}

int BackgroundSubtractor::isPersonBox(cv::Rect r1, cv::Rect r2)
{
	double intersectArea = (r1 & r2).area();
    
	if(intersectArea != 0.0)
    {
		double unionArea = (r1|r2).area();
		double overlap = intersectArea / unionArea;
		if (overlap >= 0.6) return TRUE; // || r1.area() == intersectArea
		else if (overlap > 0.1 && overlap < 0.6) return UNDEFINED;
	}
	return FALSE;
}

