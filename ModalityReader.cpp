//
//  ModalityReader.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#include "ModalityReader.h"

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::filesystem;
using namespace std;


ModalityReader::ModalityReader() : m_MasksOffset(200)
{
    
}


ModalityReader::ModalityReader(string dataPath) : m_DataPath(dataPath), m_MasksOffset(200)
{
    setDataPath(m_DataPath);
}


void ModalityReader::setDataPath(string dataPath)
{
    m_DataPath = dataPath;
    
    const char* path = m_DataPath.c_str();
	if( exists( path ) )
	{
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( is_directory( *iter ) && iter->path().string().find("Scene") != std::string::npos)
			{
                string scenePath = iter->path().string();
				m_ScenesPaths.push_back(scenePath);
                
                cout << "Scene found: " << scenePath << endl;
			}
		}
	}
    else
    {
        cerr << "Data path is not containing any scene(s)!" << endl;
    }
}


void ModalityReader::setMasksOffset(unsigned char offset)
{
    
}


void ModalityReader::read(std::string modality, ModalityData& md)
{
    vector<cv::Mat> frames;
    vector<cv::Mat> masks;
    vector<cv::Mat> gtMasks;
    vector<cv::Mat> regFrames;
    
    vector<string> framesIndices;
    
    vector< vector<cv::Rect> > rects;
    vector< vector<int> > tags;
    vector<string> calibVars;
    vector<pair<int, int> > framesPerScene;
    
    int nFrames = 0;
    for (int i = 0; i < m_ScenesPaths.size(); i++)
    {
        loadDataToMats   (m_ScenesPaths[i] + "/Frames/" + modality + "/", "jpg", frames, framesIndices);
        loadDataToMats   (m_ScenesPaths[i] + "/Masks/" + modality + "/", "png", masks);
        loadDataToMats(m_ScenesPaths[i] + "/GroundTruth/" + modality + "/", "png", gtMasks);
        loadBoundingRects(m_ScenesPaths[i] + "/Masks/" + modality + ".yml", rects, tags);
        if(modality.compare("Thermal") == 0) {
            loadCalibVarsDir (m_ScenesPaths[i] + "/calibVars.yml", calibVars);
        }
        if(modality.compare("Depth") == 0) {
            loadDataToMats(m_ScenesPaths[i] + "/Frames/" + modality + "Raw/", "jpg", regFrames);
        }
        
        framesPerScene.push_back(std::make_pair(nFrames, frames.size() - 1));
        nFrames = frames.size();
    }
    
    md.setFrames(frames); // inner copy
    md.setSceneLimits(framesPerScene);
    frames.clear(); // and clear
    
    md.setFramesIndices(framesIndices);
    framesIndices.clear();
    
    md.setPredictedMasks(masks);
    md.setMasksOffset(m_MasksOffset);
    masks.clear();
    
    md.setGroundTruthMasks(gtMasks);
    gtMasks.clear();
    
    md.setPredictedBoundingRects(rects);
    rects.clear();
    
    md.setTags(tags);
    tags.clear();
    
    if(modality.compare("Thermal") == 0) {
        md.setCalibVarsDirs(calibVars);
        calibVars.clear();
    }
    if(modality.compare("Depth") == 0) {
        md.setRegFrames(regFrames);
        regFrames.clear();
    }
}


/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void ModalityReader::loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames)
{
    const char* path = dir.c_str();
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(".png") == 0  ||
                                            iter->path().extension().string().compare(".jpg") == 0))
			{
                cout << iter->path().string() << endl; //debug
				cv::Mat img = cv::imread( iter->path().string(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
				frames.push_back(img);
            }
		}
	}
}

/**
 * Load data to opencv's cv::Mats and indices
 *
 * This method uses OpenCV and Boost.
 */
void ModalityReader::loadDataToMats(string dir, const char* format, vector<cv::Mat> & frames, vector<string>& indices)
{
    const char* path = dir.c_str();
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(".png") == 0 ||
                                            iter->path().extension().string().compare(".jpg") == 0))
			{
                cout << iter->path().string() << endl; //debug
				cv::Mat img = cv::imread( iter->path().string(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				frames.push_back(img);
                
                string filename =iter->path().filename().string();
                indices.push_back(filename.erase(filename.size()-4));
			}
		}
	}
}


/*
 * Load the people data (bounding boxes coordinates)
 */
void ModalityReader::loadBoundingRects(string file, vector< vector<cv::Rect> > & rects, vector< vector<int> >& tags)
{
    cv::FileStorage fs;
    fs.open(file.c_str(), cv::FileStorage::READ);
    
    int num_frames;
    fs["num_frames"] >> num_frames;
    
    for (int i = 0; i < num_frames; i++)
    {
        stringstream ss;
        ss << i;
        
        vector<int> v, w;
        fs[string("coords_") + ss.str()] >> v;
        fs[string("tags_") + ss.str()] >> w;

        vector<cv::Rect> frame_rects;
        for (int j = 0; j < v.size() / 4; j++)
        {
            int x0 = v[j*4];
            int y0 = v[j*4+1];
            int x1 = v[j*4+2];
            int y1 = v[j*4+3];
            
            frame_rects.push_back( cv::Rect(x0, y0, x1 - x0, y1 - y0) );
        }

        rects.push_back(frame_rects);
        tags.push_back(w);
    }
    
    fs.release();
}

void ModalityReader::loadCalibVarsDir(string dir, vector<string>& calibVarsDirs) {
    
    calibVarsDirs.push_back(dir);
    
}