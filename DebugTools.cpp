//
//  DebugTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

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
 * Visualizes in an OpenCV window the sequence of frames with the corresponding bounding boxes surrounding the items found.
 * To check wheter the bounding rects have been generated correctly.
 */
void visualizeBoundingRects(vector<cv::Mat> frames, vector<vector<cv::Rect> > rects)
{
    cv::namedWindow("Bounding rects sequence visualization");
    for (int i = 0; i < frames.size(); i++)
    {
        cout << i << " rects: " << rects[i].size() << endl;
        cv::Mat frame (frames[i]);
        
        bool any = false;
        for (int j = 0; j < rects[i].size(); j++)
        {
            cv::Rect r = rects[i][j];
            cv::rectangle(frame, cv::Point(r.x, r.y), cv::Point(r.x+r.width, r.y+r.height), cvScalar(255,0,0));
            any = true;
        }
        imshow("Bounded masks sequence visualization", frame);
        
        if (any)
            cv::waitKey();
        else
            cv::waitKey(10);
        
    }
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