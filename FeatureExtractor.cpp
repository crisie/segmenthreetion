//
//  FeatureExtractor.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{
}

void FeatureExtractor::describe(ModalityGridData& data)
{
	for (int k = 0/*2323*/; k < data.getGridsFrames().size(); k++)
	{
        if (k % 1000 == 0) cout << 100.0 * k / data.getGridsFrames().size() << "%" <<  endl; // debug
        //cout << k << endl;
        // Normal image description
        
        GridMat grid        = data.getGridFrame(k);
        GridMat gmask       = data.getGridMask(k);
        cv::Mat gvalidness  = data.getValidnesses(k);
        
        GridMat gdescriptors;
        describe(grid, gmask, gvalidness, gdescriptors);
        
        // Mirrored image description
        
        int flipCode            = 1;
        GridMat gridMirrored    = grid.flip(flipCode); // flip respect the vertical axis
        GridMat gmaskMirrored   = gmask.flip(flipCode);
        cv::Mat gvalidnessMirrored;
        cv::flip(gvalidness, gvalidnessMirrored, flipCode);
        
        GridMat gdescriptorsMirrored;
        describe(gridMirrored, gmaskMirrored, gvalidnessMirrored, gdescriptorsMirrored);
        
        // Add to the descriptors to the data
        data.addDescriptors(gdescriptors);
        data.addDescriptorsMirrored(gdescriptorsMirrored);
	}
}

/*
 * Hypercube normalization
 */
void FeatureExtractor::hypercubeNorm(cv::Mat & src, cv::Mat & dst)
{
    src.copyTo(dst);
    double z = sum(src).val[0]; // partition function :D
    dst = dst / z;
}