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
 * Visualizes in an OpenCV window the sequence of frames with the corresponding bounding boxes surrounding the items found.
 * To check wheter the bounding rects have been generated correctly.
 */
void visualizeBoundingRects(string modality, vector<cv::Mat> frames, vector<vector<cv::Rect> > rects, bool save);

/**
 * DEBUG (Auxiliary function)
 * Visualize gridmats
 */
void visualizeGridmats(vector<GridMat> gridmats);

/**
 * DEBUG (Auxiliary function)
 * Show differences per frame in number of bounding boxes between two modalities
 */
void compareNumberBoundingBoxes(vector<vector<cv::Rect> > bb1, vector<vector<cv::Rect> > bb2);

/**
 * DEBUG (Auxiliary function)
 * Add zeros to the left of a number
 */
string zeroPadNumber(int num);

#endif /* defined(__segmenthreetion__DebugTools__) */
