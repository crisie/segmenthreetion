//
//  DebugTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdarg.h>

#include "DebugTools.h"

/**
 * DEBUG (Auxiliary function)
 * Visualizes in an OpenCV window the sequence of masks surrounded by their corresponding bounding boxes.
 * To check whether the masks and bounding rects have been loaded correctly.
 */
void visualizeMasksWithinRects(vector<cv::Mat> masks, vector<vector<cv::Rect> > rects)
{
    cv::namedWindow("Bounded masks sequence visualization");
    for (int i = 0; i < masks.size(); i++)
    {
        cout << i << " rects: " << rects[i].size() << endl;
        cv::Mat gimg (masks[i]);
        cv::Mat img;
        cvtColor(gimg, img, CV_GRAY2RGB);
        bool any = false;
        for (int j = 0; j < rects[i].size(); j++)
        {
            cv::Rect r = rects[i][j];
            cv::rectangle(img, cv::Point(r.x, r.y), cv::Point(r.x+r.width, r.y+r.height), cvScalar(255,0,0));
            any = true;
        }
        imshow("Bounded masks sequence visualization", img);
        
        if (any)
            cv::waitKey();
        else
            cv::waitKey(10);
    }
}

/**
 * DEBUG (Auxiliary function)
 * Visualizes in an OpenCV window the sequence of frames with the corresponding bounding boxes surrounding the items found and save them (optional).
 * To check wheter the bounding rects have been generated correctly.
 */
void visualizeBoundingRects(string modality, vector<cv::Mat> frames, vector<vector<cv::Rect> > rects, bool save)
{
    cv::namedWindow("Bounding rects sequence visualization");
    for (int i = 0; i < frames.size(); i++)
    {
        cout << i << " rects: " << rects[i].size() << endl;
        cv::Mat frame;
        frames[i].copyTo(frame);
        bool any = false;
        for (int j = 0; j < rects[i].size(); j++)
        {
            cv::Rect r = rects[i][j];
            cv::rectangle(frame, cv::Point(r.x, r.y), cv::Point(r.x+r.width, r.y+r.height), cvScalar(255,0,0));
            any = true;
        }
        imshow("Bounded masks sequence visualization", frame);
        
        if(save) {
            std::vector<int> qualityType;
            qualityType.push_back(CV_IMWRITE_PNG_COMPRESSION);
            qualityType.push_back(3);
            stringstream aa;
            aa << "../../Sequences/" << modality << "/" << zeroPadNumber(i) << ".png";
            string filename = aa.str();
            aa.str("");
            cv::imwrite(filename,frame,qualityType);
        }
        
        if (any)
            cv::waitKey(10);
        else
            cv::waitKey(10); //10
        
    }
}

string zeroPadNumber(int num)
{
    ostringstream ss;
    ss << std::setw(5) << setfill('0') << num;
    std::string result = ss.str();
    return result;
}

/**
 * DEBUG (Auxiliary function)
 * Visualize gridmats
 */
void visualizeGridmats(vector<GridMat> gridmats)
{
    cv::namedWindow("Gridmats sequence visualization");
    for (int i = 0; i < gridmats.size(); i++)
    {
        gridmats[i].show("Gridmats sequence visualization");
        
        cv::waitKey(100);
    }
}

/**
 * DEBUG (Auxiliary function)
 * Show differences per frame in number of bounding boxes between two modalities
 */
void compareNumberBoundingBoxes(vector<vector<cv::Rect> > bb1, vector<vector<cv::Rect> > bb2)
{
    for(unsigned int i = 0;  i < bb1.size(); i++) {
        if(bb1[i].size() != bb2[i].size()) cout << "frame " << i << ": " << bb1[i].size() << " - " << bb2[i].size() << endl;
    }
}