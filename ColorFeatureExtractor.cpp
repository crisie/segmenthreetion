//
//  ColorFeatureExtractor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 13/02/14.
//
//

#include "ColorFeatureExtractor.h"
#include "ColorParametrization.hpp"


ColorFeatureExtractor::ColorFeatureExtractor()
    : FeatureExtractor()
{ }


ColorFeatureExtractor::ColorFeatureExtractor(ColorParametrization cParam)
    : FeatureExtractor(), m_ColorParam(cParam)
{ }


void ColorFeatureExtractor::setParam(ColorParametrization ColorParam)
{
    m_ColorParam = ColorParam;
}


void ColorFeatureExtractor::describeColorHog(const cv::Mat cell, const cv::Mat cellMask, cv::Mat & cOrientedGradsHist)
{
    
	/*Following R-HoG descriptor (Dalal-Triggs 2005) except that the image patch is variance-normalized before
     binning into the histogram, making the descriptor less sensitive to illumination changes.
     
     Basic pipeline:
     1. Variance normalize the image patch
     2. Convolve the image patch with k = [-1 0 1] both horizontally and vertically
     3. Subdivide the image patch into cells
     4. For each cell compute the unsigned or signed gradient angle (unsigned seems to work better) for
     all the pixels in the cell as well as some neighbouring cells (determined by the block size).
     - Accumulate local histogram “energy” over a larger regions (“blocks”) to normalize
     all of the cells in the block.
     5. Bin the gradient magnitude into a bin based on the computational angle, weighted by a gaussian
     kernel centered on the cell's center (thus pixels further away get less weight).
     6. Normalize that cell's histogram
     */
    
    cv::Size gridSize = cv::Size(m_ColorParam.winSizeX,m_ColorParam.winSizeY);
    
    cv::Mat tmpCell, tmpCellMask;
	resize(cell, tmpCell, gridSize);
	resize(cellMask, tmpCellMask, gridSize);
    
    int cellSizeX = m_ColorParam.cellSizeX; //cellgridsize
    int cellSizeY = m_ColorParam.cellSizeY;
    int blockSizeX = m_ColorParam.blockSizeX;
    int blockSizeY = m_ColorParam.blockSizeY;
    int hogbins = m_ColorParam.nbins;
    
    int nBlocksX = tmpCell.rows/blockSizeX;
    int nBlocksY = tmpCell.cols/blockSizeY;
    
    int nCellsX = blockSizeX/cellSizeX;
    int nCellsY = blockSizeY/cellSizeY;
    
    int lengthDescriptor = hogbins * nCellsX * nCellsY * nBlocksX * nBlocksY;
    
	vector<cv::Mat> rgbCell;
	split(tmpCell,rgbCell); //b,g,r
    
	vector<cv::Mat> maskCellDervX(3), maskCellDervY(3), cellGradOrients(3);
	for(unsigned int c = 0; c < 3; c++) {
	    // First derivatives
        cv::Mat cellDervX, cellDervY;
        
        //Compute gradients with [-1,0,1] kernel
		Sobel(rgbCell[c], cellDervX, CV_32F, 1, 0);
		Sobel(rgbCell[c], cellDervY, CV_32F, 0, 1);
        
		//Apply masks
        maskCellDervX[c] = cv::Mat::zeros(cellDervX.rows, cellDervX.cols, cellDervX.type());
        maskCellDervY[c] = cv::Mat::zeros(cellDervY.rows, cellDervY.cols, cellDervY.type());
		cellDervX.copyTo(maskCellDervX[c], tmpCellMask);
		cellDervY.copyTo(maskCellDervY[c], tmpCellMask);
        
		// Compute gradient angles
		phase(maskCellDervX[c], maskCellDervY[c], cellGradOrients[c], true);
	}
    
    cv::Mat tmpHist(0,lengthDescriptor, cv::DataType<float>::type);
    
    //Subdivide the image patch into blocks
    for (int b_r = 0; b_r < cellGradOrients[1].rows; b_r+=blockSizeX)
    	for (int b_c = 0; b_c < cellGradOrients[1].cols; b_c+=blockSizeY)
        {
            cv::Mat block = cellGradOrients[1](cv::Range(b_r, min(b_r+blockSizeX, cellGradOrients[1].rows)),
                                               cv::Range(b_c, min(b_c+blockSizeY, cellGradOrients[1].cols))).clone();
            //    	imshow("block", block);
            cv::Mat cellHist = cv::Mat::zeros(nCellsX*nCellsY, hogbins, cv::DataType<float>::type);
            // Subdivide block into cells
            int nCell = 0;
            for (int c_r = 0; c_r < block.rows; c_r+=cellSizeX) for (int c_c = 0; c_c < block.cols; c_c+=cellSizeY)
            {
                cv::Mat tile = block(cv::Range(c_r, min(c_r+cellSizeX, cellGradOrients[1].rows)),
                                     cv::Range(c_c, min(c_c+cellSizeY, cellGradOrients[1].cols))).clone();
                //    		imshow("tile", tile);
                
                for (int i = b_r+c_r; i < b_r+c_r+tile.rows; i++) for (int j = b_c+c_c; j < b_c+c_c+tile.cols; j++)
                {
                    float magnitudeTemp = -0.1;
                    int channel = -1;
                    for(unsigned int c = 0; c < 3; c++) {
                        float g_x = maskCellDervX[c].at<unsigned short>(i,j);
                        float g_y = maskCellDervY[c].at<unsigned short>(i,j);
                        
                        float magnitude = sqrtf(g_x * g_x + g_y * g_y);
                        
                        if(magnitude > magnitudeTemp)  {
                            magnitudeTemp = magnitude;
                            channel = c;
                        }
                    }
                    
                    float g_x = maskCellDervX[channel].at<unsigned short>(i,j);
                    float g_y = maskCellDervY[channel].at<unsigned short>(i,j);
                    
                    float orientation = cellGradOrients[channel].at<float>(i,j);
                    if(orientation == 360.0) {
                        cout << orientation << endl;
                    }
                    if(orientation > 180.0) orientation = orientation - 180.0;
                    float bin = static_cast<int>((orientation/180.0) * hogbins) % hogbins;
                    cellHist.at<float>(nCell, bin) += sqrtf(g_x * g_x + g_y * g_y);
                    //                    cout << "orientation : " << orientation << endl;
                    //                    cout << "magnitude : " << cellHist.at<float>(nCell, bin) << endl;
                }
                nCell++;
                tile.release();
                //    		destroyWindow("tile");
            }
            
            cv::Mat normCellHist;
            cv::Mat vCellHist = cellHist.reshape(0,1).clone();
            normalize(vCellHist, normCellHist);
            tmpHist.push_back(normCellHist.row(0));
            
            //            cout << "normalized block dimensions: " << normCellHist.cols << " width x " << normCellHist.rows << "height" << endl;
            //            cout << "tmpHist: " << tmpHist.cols << " width x " << tmpHist.rows << "height" << endl;
            
            vCellHist.release();
            normCellHist.release();
            block.release();
            cellHist.release();
            //    	destroyWindow("block");
        }
    //    destroyWindow("cellGradOrients");
    //    destroyWindow("maskCellDervX");
    //    destroyWindow("maskCellDervY");
    //    destroyWindow("cellDervX");
    //    destroyWindow("cellDervY");
    //    destroyWindow("Cell");
    //    destroyWindow("CellMask");
    //    destroyWindow("Gray Cell");
    cv::Mat vTmpHist = tmpHist.reshape(0,1).clone();
    hypercubeNorm(vTmpHist, cOrientedGradsHist);
    //Mat visu = get_hogdescriptor_visu(tmpCell, tmpCellMask, descriptor);
    //imshow("hog visualization", visu);
    //if(waitKey(30)>= 0)  {}
    //cout << "descriptor dimensions: " << tOrientedGradsHist.cols << " width x " << tOrientedGradsHist.rows << "height" << endl;
	//cout << "Found " << tOrientedGradsHist.size() << " descriptor values" << endl;
    vTmpHist.release();
	tmpHist.release();
}


void ColorFeatureExtractor::describe(ModalityGridData data, GridMat & descriptors)
{
	for (int k = 0; k < data.getGridsFrames().size(); k++)
	{
		cout << "k : " << k <<  endl;
        
        GridMat grid = data.getGridFrame(k);
        GridMat gmask = data.getGridMask(k);
        
        for(int i = 0; i < grid.crows(); i++) for (int j = 0; j < grid.ccols(); j++)
        {
            cv::Mat & cell = grid.at(i,j);
            cv::Mat & tmpCellMask = gmask.at(i,j);
            cv::Mat cellMask = cv::Mat::zeros(tmpCellMask.rows, tmpCellMask.cols, CV_8UC1);
            cvtColor(tmpCellMask, tmpCellMask, CV_RGB2GRAY);
            threshold(tmpCellMask,tmpCellMask,1,255,CV_THRESH_BINARY);
            tmpCellMask.convertTo(cellMask, CV_8UC1);
            
            //HOG descriptor
            cv::Mat cOrientedGradsHist;
            describeColorHog(cell, cellMask, cOrientedGradsHist);
            
            descriptors.vconcat(cOrientedGradsHist, i, j);
        }
	}
}


cv::Mat ColorFeatureExtractor::get_hogdescriptor_visu(cv::Mat origImg, cv::Mat mask, vector<float> descriptorValues)
{
    cv::Mat color_origImg;
    origImg.copyTo(color_origImg, mask);
    cvtColor(color_origImg, color_origImg, CV_GRAY2RGB);
    
    float zoomFac = 3;
    cv::Mat visu;
    cv::resize(color_origImg, visu, cv::Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));
    
    int blockSize       = 32;
    int cellSize        = 16;
    int gradientBinSize = 9;
    float radRangeForOneBin = M_PI/(float)gradientBinSize; // dividing 180° into 9 bins, how large (in rad) is one bin?
    
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = color_origImg.cols / cellSize;
    int cells_in_y_dir = color_origImg.rows / cellSize;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
            
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
    
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
    
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
    
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
                
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
                    
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                    
                } // for (all bins)
                
                
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
                
            } // for (all cells)
            
            
        } // for (all block x pos)
    } // for (all block y pos)
    
    
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
            
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
    
    
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
    
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
            
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
            
            cv::rectangle(visu, cv::Point(drawX*zoomFac,drawY*zoomFac), cv::Point((drawX+cellSize)*zoomFac,(drawY+cellSize)*zoomFac), CV_RGB(100,100,100), 1);
            
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
                
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
                
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
                
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize/2;
                float scale = 2.5; // just a visualization scale, to see the lines better
                
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                
                // draw gradient visualization
                cv::line(visu, cv::Point(x1*zoomFac,y1*zoomFac), cv::Point(x2*zoomFac,y2*zoomFac), CV_RGB(0,255,0), 1);
                
            } // for (all bins)
            
        } // for (cellx)
    } // for (celly)
    
    
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    
    cout << "END" << endl;
    return visu;
    
} // get_hogdescriptor_visu

