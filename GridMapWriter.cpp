//
//  GridMapWriter.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 07/03/14.
//
//

#include "GridMapWriter.h"

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <set>

using namespace boost::filesystem;


// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void GridMapWriter::write<unsigned char>(std::string dir);
template void GridMapWriter::write<int>(std::string dir);
template void GridMapWriter::write<float>(std::string dir);
template void GridMapWriter::write<double>(std::string dir);

template void GridMapWriter::write<unsigned char>(ModalityGridData& mgd, GridMat& values, std::string dir);
template void GridMapWriter::write<int>(ModalityGridData& mgd, GridMat& values, std::string dir);
template void GridMapWriter::write<float>(ModalityGridData& mgd, GridMat& values, std::string dir);
template void GridMapWriter::write<double>(ModalityGridData& mgd, GridMat& values, std::string dir);
// -----------------------------------------------------------------------------


GridMapWriter::GridMapWriter()
{
    
}

GridMapWriter::GridMapWriter(ModalityGridData& mgd, GridMat& values)
: m_mgd(mgd), m_values(values)
{
    
}

void GridMapWriter::setModalityGridData(ModalityGridData& mgd)
{
    m_mgd = mgd;
}

void GridMapWriter::setGridCellValues(GridMat& values)
{
    m_values = values;
}

template<typename T>
void GridMapWriter::write(std::string outputDir)
{
    write<T>(m_mgd, m_values, outputDir);
}

unsigned char colors[][3] = {{255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}};
template<typename T>
void GridMapWriter::write(ModalityGridData& mgd, GridMat& gvalues, string outputDir)
{
    cv::Mat gridsScenesIDs = mgd.getSceneIDsMat();
    cv::Mat gridsFramesIDs = mgd.getFrameIDsMat();
    
    cv::Mat counts (mgd.getHp(), mgd.getWp(), cv::DataType<int>::type);
    counts.setTo(0);
    
    for (int s = 0; s < mgd.getNumOfScenes(); s++)
    {
        cv::Mat framesInSceneIndices;
        cv::findNonZero(gridsScenesIDs == s, framesInSceneIndices);
        
        int begin   = framesInSceneIndices.at<int>(0,1);
        int end     = framesInSceneIndices.at<int>(framesInSceneIndices.rows-1, 1);
        
        set<int> e (mgd.getFrameIDs().begin() + begin,
                    mgd.getFrameIDs().begin() + end);
        vector<int> uniqueFrameIDs (e.begin(), e.end());
        
        for (int f = 0; f < uniqueFrameIDs.size(); f++)
        {
            int fid = uniqueFrameIDs[f];
            
            cv::Mat gridsInFrameIndices, aux;
            cv::bitwise_and(gridsScenesIDs == s, gridsFramesIDs == fid, aux);
            cv::findNonZero(aux, gridsInFrameIndices);

            int idx0 = gridsInFrameIndices.at<int>(0,1);
            
            string frameFilename = mgd.getFrameFilename(idx0);
            string frameFilePath = mgd.getFramePath(idx0);
            cv::Point2d res = mgd.getFrameResolution(idx0);
            
            string modality = mgd.getModality();
            if (modality.compare("Motion") == 0 || modality.compare("Ramanan") == 0)
            {
                modality = "Color";
            }
            cv::Mat mask = cv::imread(frameFilePath + "Masks/" + modality + "/" + frameFilename + ".png",
                                      CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            
//            // DEBUG
//            // +----------------------------------------------------------------
//            cv::Mat drawableMask;
//            cv::cvtColor(mask.clone(), drawableMask, CV_GRAY2BGR);
//            // +----------------------------------------------------------------
            
            cv::Mat map (res.y, res.x, cv::DataType<T>::type);
            map.setTo(0);
            
            for (int k = 0; k < gridsInFrameIndices.rows; k++)
            {
                cout << "scene " << s << ", filename " << frameFilename << ", rect " << k << "/" << gridsInFrameIndices.rows << endl;
                int idx = gridsInFrameIndices.at<int>(k,1);
                
                cv::Rect r = mgd.getGridBoundingRect(idx);
                cv::Mat validnesses = mgd.getValidnesses(idx).clone();
                
//                // DEBUG
//                // +----------------------------------------------------------------
//                unsigned char red, green, blue;
//                red = colors[k][0];
//                green = colors[k][1];
//                blue = colors[k][2];
//                cv::rectangle(drawableMask, cv::Point(r.x, r.y), cv::Point(r.x+r.width, r.y+r.height),
//                              cv::Scalar(red,green,blue,0),1,8,0);
//                // +----------------------------------------------------------------
                
                cv::Mat roiMap (map, r);
                GridMat gRoiMap (roiMap, mgd.getHp(), mgd.getWp());

                cv::Mat roiMask (mask, r);
                cv::Mat indexedmaskroi;
                roiMask.copyTo(indexedmaskroi, roiMask == mgd.getGridMaskOffset(idx));
                GridMat gRoiMask (indexedmaskroi, mgd.getHp(), mgd.getWp());

                for (int i = 0; i < mgd.getHp(); i++) for (int j = 0; j < mgd.getWp(); j++)
                {
                    T value = gvalues.at<T>(i, j, counts.at<int>(i,j)++, 0) * 255;
                    gRoiMap.setTo(value, i, j, gRoiMask.at(i,j));
                }
                
                gRoiMap.convertToMat<T>().copyTo(roiMap, roiMask);
            }
            
            string mapPath = frameFilePath + "Maps/" + outputDir + frameFilename + ".png";
            cv::imwrite(mapPath, map);
            
//            // DEBUG
//            // +----------------------------------------------------------------
//            cv::Mat drawableMap;
//            cv::cvtColor(map, drawableMap, CV_GRAY2BGR);
//            for (int k = 0; k < gridsInFrameIndices.rows; k++)
//            {
//                cout << frameFilename << ", " << k << "/" << gridsInFrameIndices.rows << endl;
//                int idx = gridsInFrameIndices.at<int>(k,1);
//                
//                cv::Rect r = mgd.getGridBoundingRect(idx);
//                unsigned char red, green, blue;
//                red = colors[k][0];
//                green = colors[k][1];
//                blue = colors[k][2];
//                cv::rectangle(drawableMap, cv::Point(r.x, r.y), cv::Point(r.x+r.width, r.y+r.height),
//                              cv::Scalar(red,green,blue,0),1,8,0);
//            }
//            
//            cv::namedWindow("a");
//            cv::imshow("a", drawableMap);
//            cv::waitKey();
//            // +----------------------------------------------------------------
            
        }
    }
}


/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void GridMapWriter::loadFilenames(string dir, const char* filetype, vector<string>& filenames)
{
	filenames.clear();
    
    const char* path = dir.c_str();
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
            string extension = "." + string(filetype);
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(extension) == 0) )
			{
				string filename = iter->path().filename().string();
				filenames.push_back(filename.substr(0,filename.size()-4));
            }
		}
	}
}
