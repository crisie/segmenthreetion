//
//  MotionFeatureExtractor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#include "MotionFeatureExtractor.h"

#include <opencv2/video/video.hpp>


MotionFeatureExtractor::MotionFeatureExtractor()
    : FeatureExtractor()
{}


MotionFeatureExtractor::MotionFeatureExtractor(MotionParametrization param)
    : FeatureExtractor(), m_Param(param)
{}


void MotionFeatureExtractor::setParam(MotionParametrization param)
{
    m_Param = param;
}


void MotionFeatureExtractor::describe(ModalityGridData data, GridMat& descriptors)
{
	for(int k = 0; k < data.getGridsFrames().size(); k++)
	{
		cout << "k : " << k <<  endl;
        
        GridMat grid = data.getGridFrame(k);
        GridMat gmask = data.getGridMask(k);
        
        for (int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.ccols(); j++)
        {
            cv::Mat & cell = grid.at(i,j);
            cv::Mat & tmpCellMask = gmask.at(i,j);
            cv::Mat cellMask = cv::Mat::zeros(tmpCellMask.rows, tmpCellMask.cols, CV_8UC1);
            cvtColor(tmpCellMask, tmpCellMask, CV_RGB2GRAY);
            threshold(tmpCellMask,tmpCellMask,1,255,CV_THRESH_BINARY);
            tmpCellMask.convertTo(cellMask, CV_8UC1);
            
            cv::Mat mOrientedFlowHist;
            describeMotionOrientedFlow(cell, cellMask, mOrientedFlowHist);
            
            descriptors.vconcat(mOrientedFlowHist, i, j);
        }
	}
}


void MotionFeatureExtractor::describeMotionOrientedFlow(const cv::Mat cell, const cv::Mat cellMask, cv::Mat & tOrientedFlowHist)
{
	int ofbins = m_Param.hoofbins;
	//imshow("cell", cell);
    cv::Mat cellSeg = cv::Mat::zeros(cell.rows, cell.cols, cell.type());
	cell.copyTo(cellSeg, cellMask);
	//imshow("cellSeg", cellSeg);
    cv::Mat tmpHist = cv::Mat::zeros(1, ofbins, cv::DataType<float>::type);
    
    cv::Mat cellGradOrients = cv::Mat::zeros(cellSeg.rows, cellSeg.cols, CV_64FC1);
	//Mat cellSegX = cellSeg.at<Point2f>().x;
	//Mat cellSegY = cellSeg.at<Point2f>().y;
	vector<cv::Mat> comps(2);
	split(cellSeg, comps);
    //	if(comps[0].size() != comps[1].size()) {
    //		cout << "no tengo la misma size" << endl;
    //	}
    //	if(comps[1].type() != cellGradOrients.type()) {
    //		cout << "no tengo el mismo tipo" << endl;
    //	}
    //	if(cellGradOrients.depth() != CV_32F || cellGradOrients.depth() != CV_64F) {
    //		cout << "no tengo la depth apropiada" << endl;
    //	}
	phase(comps[0], comps[1], cellGradOrients, true);
	//imshow("comps 0", comps[0]);
	//imshow("comps 1", comps[1]);
	//imshow("cellGradOrients", cellGradOrients);
    
	for (int i = 0; i < cellSeg.rows; i++) for (int j = 0; j < cellSeg.cols; j++)
	{
        cv::Point2f fxy = cellSeg.at<cv::Point2f>(i,j);
		float g_x = fxy.x;
		float g_y = fxy.y;
        
		float orientation = cellGradOrients.at<float>(i,j);
		float bin = static_cast<int>((orientation/360.0) * ofbins) % ofbins;
		tmpHist.at<float>(0, bin) += sqrtf(g_x * g_x + g_y * g_y);
		//cout << "orientation : " << orientation << endl;
		//cout << "magnitude : " << tmpHist.at<float>(0, bin) << endl;
	}
    
    cellSeg.release();
    cellGradOrients.release();
    
    //destroyWindow("cell");
    //destroyWindow("cellSeg");
    //destroyWindow("comps 0");
    //destroyWindow("comps 1");
    //destroyWindow("cellGradOrients");
    hypercubeNorm(tmpHist, tOrientedFlowHist);
    tmpHist.release();
    
}


void MotionFeatureExtractor::computeOpticalFlow(vector<cv::Mat> colorFrames, vector<cv::Mat> & motionFrames)
{
	cv::Mat frame, prev_frame;
	//Mat tmpFrame, tmpPrev_frame;
    
	vector<cv::Mat> tempColorFrames = colorFrames;
	tempColorFrames.push_back(colorFrames[colorFrames.size() - 1]);
    
	//CHANGE!!!
	double pyr_scale = 0.5;
	int levels = 3;
    int winsize = 2;
	//int winsize = 15;
	int iterations = 3;
	int poly_n = 5;
    double poly_sigma = 1.1;
	//double poly_sigma = 1.2;
	int flags = 0;
	//for(int it = 250; it < 500; it++) {
	for(int it = 0;it < tempColorFrames.size(); it++) {
        cout << it << endl;
		cv::Mat tmpFrame, tmpPrev_frame;
		frame.release();
		tempColorFrames[it].copyTo(frame);
		//imshow("current frame", frame);
		if(prev_frame.empty())
		{
			frame.copyTo(prev_frame);
		} else {
			if(!frame.empty() && !prev_frame.empty()) {
                //imshow("previous frame", prev_frame);
                
                cvtColor(frame, tmpFrame, CV_RGB2GRAY);
                cvtColor(prev_frame, tmpPrev_frame, CV_RGB2GRAY);
                
                //optical flow from previous frame to current frame (forward)
                cv::Mat flow;
                calcOpticalFlowFarneback(tmpPrev_frame,tmpFrame, flow, pyr_scale, levels,
                                         winsize, iterations, poly_n, poly_sigma, flags);
                
                motionFrames.push_back(flow);
                flow.release();
			}
			prev_frame.release();
			frame.copyTo(prev_frame);
		}
		if(cv::waitKey(30)>= 0)  {break;}
		tmpFrame.release();
		tmpPrev_frame.release();
	}
	//flow vector would have motionFrames.size()-1, so add a last flow frame = 0 ??????
	//Mat last_flow  (frame.rows, frame.cols, frame.depth());
    
	//last_flow.release();
	frame.release();
	prev_frame.release();
}
