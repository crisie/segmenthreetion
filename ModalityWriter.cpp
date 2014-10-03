//
//  ModalityWriter.cpp
//  segmenthreetion
//
//  Created by Cristina Palmero Cantariño on 10/03/14.
//
//

#include "ModalityWriter.h"

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::filesystem;
using namespace std;

ModalityWriter::ModalityWriter()
{ }


ModalityWriter::ModalityWriter(string dataPath) : m_DataPath(dataPath)
{
    setDataPath(m_DataPath);
}


void ModalityWriter::setDataPath(string dataPath)
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

void ModalityWriter::write(string modality, ModalityData& md)
{
    for (int i = 0; i < m_ScenesPaths.size(); i++)
    {
        saveMats(m_ScenesPaths[i] + "/Masks/" + modality + "/",
                 ".png",
                 *md.getPredictedMasksInScene(i),
                 *md.getFramesIndicesInScene(i));
        
        saveMats(m_ScenesPaths[i] + "/GroundTruth/" + modality + "/",
                 ".png",
                 *md.getGroundTruthMasksInScene(i),
                 *md.getFramesIndicesInScene(i));
        saveBoundingRects(m_ScenesPaths[i] + "/Masks/" + modality + ".yml",
                          *md.getPredictedBoundingRectsInScene(i),
                          *md.getTagsInScene(i));

    }
    
}

void ModalityWriter::saveMats(string dir, const char *format, vector<cv::Mat> frames, vector<string> filenames)
{
    std::vector<int> qualityType;
    qualityType.push_back(CV_IMWRITE_PNG_COMPRESSION);
    qualityType.push_back(3);

    for(int f = 0; f < frames.size(); f++)
    {
        stringstream fn;
        fn << dir << filenames[f] << format;
        string fileName = fn.str();
        
        imwrite(fileName,frames[f],qualityType);
    }
    
}

template<typename T>
void ModalityWriter::serialize(vector<vector<T> > m, vector<T>& s)
{
    s.clear();
    for (int i = 0; i < m.size(); i++)
    {
        s.reserve(s.size() + m[i].size());
        s.insert(s.end(), m[i].begin(), m[i].end());
    }
}

void ModalityWriter::saveValidBoundingBoxes(string file, vector<vector<int> > validBbs)
{
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    
    //vector<int> srlValidBbs;
    //serialize(validBbs, srlValidBbs);
    
    //fs << "validBbs" << validBbs;
    fs << "num_frames" << static_cast<int>(validBbs.size());
    
    for(unsigned int i=0;i<validBbs.size();i++)
    {
        fs << ("frame_" + to_string(i)) << validBbs.at(i);
    }
    
    fs.release();
}

void ModalityWriter::loadValidBoundingBoxes(string file, vector<vector<int> >& validBbs)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    int n;
    fs["num_frames"] >> n;
    validBbs.resize(n);
    for(unsigned int i=0;i<validBbs.size();i++)
    {
        fs["frame_" + to_string(i)] >> validBbs[i];
    }
    
    fs.release();
}

void ModalityWriter::loadValidBoundingBoxes(string file, vector<int> & validBbs)
{
    vector<vector<int> > _validBbs;
    
    loadValidBoundingBoxes(file, _validBbs);
    serialize(_validBbs, validBbs);
}

void ModalityWriter::saveBoundingRects(string file, vector<vector<cv::Rect> > rects, vector<vector<int> > tags)
{
    vector<vector<int> >  rectsInt(rects.size());
    this->boundRectsToInt(rects, rectsInt);
    
    cv::FileStorage fs(file, cv::FileStorage::WRITE);

    fs << "num_frames" << static_cast<int>(rectsInt.size());
    
    for(unsigned int i=0;i<rectsInt.size();i++)
    {
        stringstream ss;
        string s;
        ss << i;
        s = ss.str();
        string a = "coords_";
        a+=s;
        fs << a << rectsInt.at(i);
    }
    
    for(unsigned int i=0;i<tags.size();i++)
    {
        stringstream ss;
        string s;
        ss << i;
        s = ss.str();
        string a = "tags_";
        a+=s;
        fs << a << tags.at(i);
    }
    
    fs.release();
}

void ModalityWriter::boundRectsToInt(vector<vector<cv::Rect> > bbModal, vector<vector<int> >& bbModalInt) {
    
    for(unsigned int d = 0; d < bbModal.size() ; d++) //frames
    {
        for(unsigned int r = 0; r < bbModal[d].size(); r++)
        {
            bbModalInt[d].push_back(bbModal[d][r].tl().x);
            bbModalInt[d].push_back(bbModal[d][r].tl().y);
            bbModalInt[d].push_back(bbModal[d][r].br().x);
            bbModalInt[d].push_back(bbModal[d][r].br().y);
            
        }
    }
}