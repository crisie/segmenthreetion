//
//  DebugTools.h
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#ifndef __segmenthreetion__DebugTools__
#define __segmenthreetion__DebugTools__

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GridMat.h"

using namespace std;

/**
 * DEBUG (Auxiliary function)
 * Visualizes in an OpenCV window the sequence of masks surrounded by their corresponding bounding boxes.
 * To check whether the masks and bounding rects have been loaded correctly.
 */
void visualizeMasksWithinRects(vector<cv::Mat> masks, vector<vector<cv::Rect> > rects);

/**
 * DEBUG (Auxiliary function)
 * Visualize gridmats
 */
void visualizeGridmats(vector<GridMat> gridmats);


#endif /* defined(__segmenthreetion__DebugTools__) */
