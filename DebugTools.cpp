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