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

#include "GridMat.h"
#include "MotionFeatureExtractor.h"
#include "StatTools.h"


ModalityReader::ModalityReader() : m_MasksOffset(200), m_MaxOffset(8)
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
			if ( is_directory( *iter ) && iter->path().string().find("Scene") != string::npos)
			{
                cout << "Loading scene ... " << endl;
                m_ScenesPaths.push_back(iter->path().string() + "/");
                cout << "Path: " << m_ScenesPaths.back() << endl;
            
                cv::FileStorage fs (m_ScenesPaths.back() + "Partition.yml", cv::FileStorage::READ);
                if (fs.isOpened())
                    cout << "Partition file: FOUND" << endl;
                    
                cv::Mat sceneFramesPartition;
                sceneFramesPartition.push_back(fs["partition"]);
                
                cout << endl;
			}
        }
	}
    else
    {
        cerr << "Data path is not containing any scene(s)!" << endl;
    }
}

unsigned int ModalityReader::getNumOfScenes()
{
    return m_ScenesPaths.size();
}

string ModalityReader::getScenePath(unsigned int sid)
{
    return m_ScenesPaths[sid];
}


cv::Mat ModalityReader::getScenePartition(unsigned int sid)
{
    cv::Mat partition;
    cv::FileStorage fs (m_ScenesPaths[sid] + "Partition.yml", cv::FileStorage::READ);
    fs["partition"] >> partition;
    fs.release();

    return partition;
}

vector<cv::Mat> ModalityReader::getPartitions()
{
    vector<cv::Mat> scenesPartitions;
    for (int s = 0; s < m_ScenesPaths.size(); s++)
    {
        cv::Mat partition;
        cv::FileStorage fs (m_ScenesPaths[s] + "Partition.yml", cv::FileStorage::READ);
        fs["partition"] >> partition;
        fs.release();
        scenesPartitions.push_back(partition);
    }

    return scenesPartitions;
}

cv::Mat ModalityReader::getAllScenesPartition()
{
    cv::Mat merge;
    for (int s = 0; s < m_ScenesPaths.size(); s++)
    {
        cv::Mat partition;
        cv::FileStorage fs (m_ScenesPaths[s] + "Partition.yml", cv::FileStorage::READ);
        fs["partition"] >> partition;
        fs.release();
        if (merge.empty()) merge = partition;
        else cv::vconcat(merge, partition, merge);
    }
    
    return merge;
}

void ModalityReader::setMasksOffset(unsigned char offset)
{
    m_MasksOffset = offset;
}


void ModalityReader::read(string modality, ModalityData& md)
{
    vector<cv::Mat> frames;
    vector<cv::Mat> masks;
    vector<cv::Mat> gtMasks;
    vector<cv::Mat> regFrames;
    
    vector<string> filenames;
    
    vector< vector<cv::Rect> > rects;
    vector< vector<int> > tags;
    vector<string> calibVars;
    vector<pair<int, int> > framesPerScene;
    
    int nFrames = 0;
    for (int i = 0; i < m_ScenesPaths.size(); i++)
    {
        if(modality.compare("Depth") == 0)
            loadDataToMats   (m_ScenesPaths[i] + "Frames/" + modality + "/", "png", frames, filenames);
        else
            loadDataToMats   (m_ScenesPaths[i] + "Frames/" + modality + "/", "jpg", frames, filenames);

        loadDataToMats   (m_ScenesPaths[i] + "Masks/" + modality + "/", "png", masks);
        loadDataToMats(m_ScenesPaths[i] + "GroundTruth/" + modality + "/", "png", gtMasks);
        loadBoundingRects(m_ScenesPaths[i] + "Masks/" + modality + ".yml", rects, tags);
        if(modality.compare("Thermal") == 0) {
            loadCalibVarsDir (m_ScenesPaths[i] + "calibVars.yml", calibVars);
        }
        if(modality.compare("Depth") == 0) {
            loadDataToMats(m_ScenesPaths[i] + "Frames/" + modality + "Raw/", "png", regFrames);
        }
        
        framesPerScene.push_back(std::make_pair(nFrames, frames.size() - 1));
        nFrames = frames.size();
    }
    
    md.setModality(modality);
    md.setFrames(frames); // inner copy
    md.setSceneLimits(framesPerScene);
    frames.clear(); // and clear
    
    md.setFramesIndices(filenames);
    filenames.clear();
    
    md.setPredictedMasks(masks);
    md.setMasksOffset(m_MasksOffset);
    masks.clear();
    
    md.setGroundTruthMasks(gtMasks);
    gtMasks.clear();
    
    md.setPredictedBoundingRects(rects);
    rects.clear();
    
    md.setTags(tags);
    tags.clear();
    
    if(modality.compare("Thermal") == 0)
    {
        md.setCalibVarsDirs(calibVars);
        calibVars.clear();
    }
    
    if(modality.compare("Depth") == 0)
    {
        md.setRegFrames(regFrames);
        regFrames.clear();
    }
}

void ModalityReader::readAllScenesData(string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    mgd.clear();
    mgd.setModality(modality);
    mgd.setHp(hp);
    mgd.setWp(wp);
    
    for (int s = 0; s < m_ScenesPaths.size(); s++)
    {
        readSceneData(s, modality, filetype, hp, wp, mgd);
    }
}

void ModalityReader::readSceneData(unsigned int sid, string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    if (mgd.getModality().compare("") == 0)
        mgd.setModality(modality);
    else
        assert(mgd.getModality().compare(modality) == 0);
    
    if (mgd.getHp() == 0)
        mgd.setHp(hp);
    else
        assert(mgd.getHp() == hp);
    
    if (mgd.getWp() == 0)
        mgd.setWp(wp);
    else
        assert(mgd.getWp() == wp);
    
	// auxiliary
	vector<string> framesFilenames; // Frames' filenames from <dataDir>/Frames/<modality>/
    vector<string> masksFilenames; // Corresponding masks' filenames
	vector<vector<cv::Rect> > rects; // Bounding rects at frame level (having several per frame)
	vector<vector<int> > tags; // Tags corresponding to the bounding rects
    
    cout << "Searching for " << modality << " files ... ";
    
    // Motion & Ramanan modalities need special treatment for reading
    // .. In motion case, the optical flow vectors must be computed for every pair of frames
    // .. In ramanan, the frames are already computed probability maps (from Matlab)
    if (modality.compare("Motion") == 0)
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Frames/Color/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/Color.yml", rects, tags);
    }
    else if (modality.compare("Ramanan") == 0)
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Maps/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/Color.yml", rects, tags);
    }
    else
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Frames/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/" + modality + ".yml", rects, tags);
    }
    
    assert (framesFilenames.size() == masksFilenames.size());
    
    cout << "DONE" << endl;
    
    cout << "Loading validation partitions ... ";
    
    cv::Mat partition = getScenePartition(sid);
    
    cout << "DONE" << endl;
    
    // Load frame-wise (Mat), extract the roi represented by the bounding boxes,
    // grid the rois (GridMat), and store in GridModalityData object
    
    cout << "Loading and griding frames and masks ... " << endl;
    
    // ***
    cv::Mat prevFrame; // used in motion modality
    // ***
    
	for (int f = 0; f < framesFilenames.size(); f++)
	{
        if (rects[f].size() < 1)
            continue;
        
        // Load the frame and its mask
        string framePath, maskPath;
        
        if (modality.compare("Motion") == 0)
        {
            framePath = m_ScenesPaths[sid] + "Frames/Color/" + framesFilenames[f] + "." + filetype;
            maskPath  = m_ScenesPaths[sid] + "Masks/Color/" + masksFilenames[f] + ".png";
        }
        else if (modality.compare("Ramanan") == 0)
        {
            framePath = m_ScenesPaths[sid] + "Maps/Ramanan/" + framesFilenames[f] + "." + filetype;
            maskPath  = m_ScenesPaths[sid] + "Masks/Color/" + masksFilenames[f] + ".png";
        }
        else
        {
            framePath = m_ScenesPaths[sid] + "Frames/" + modality + "/" + framesFilenames[f] + "." + filetype;
            maskPath  = m_ScenesPaths[sid] + "Masks/" + modality + "/" + masksFilenames[f] + ".png";
        }
        
        cv::Mat frame;
        if (modality.compare("Ramanan") != 0)
            frame = cv::imread(framePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        else
            frame = cvx::matlabread<double>(framePath); // ramanan maps are matlab matrices of doubles
        
		cv::Mat mask  = cv::imread(maskPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

        // (Motion modality) load also a second color frame to compute the actual motion frame
        // --------------------------------------------------------------------------------------
        if (modality.compare("Motion") == 0)
        {
            cv::Mat currFrame = cv::imread(m_ScenesPaths[sid] + "Frames/Color/" + framesFilenames[f] + "." + filetype, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);;
            
            if (prevFrame.empty()) currFrame.copyTo(prevFrame);
            
            MotionFeatureExtractor::computeOpticalFlow(pair<cv::Mat,cv::Mat>(prevFrame,currFrame), frame);
            
            currFrame.copyTo(prevFrame);
        }
        // --------------------------------------------------------------------------------------
        
        // Look the bounding rects in it...

		for (int r = 0; r < rects[f].size(); r++)
		{
//            cout << "+ frame " << f << ", grid " << r << " : ";
            // Create the grid structures
            
			if (rects[f][r].height >= hp && rects[f][r].width >= wp)
			{
				cv::Mat subjectroi (frame, rects[f][r]); // Get a roi in frame defined by the rectangle.
				GridMat gsubject (subjectroi, hp, wp);
				mgd.addGridFrame( gsubject );
                
//                cout << "F";
                
                double minFVal, maxFVal;  // keep global min and max normalization purposes
                cv::minMaxIdx(subjectroi, &minFVal, &maxFVal);
                if (minFVal < mgd.getMinVal()) mgd.setMinVal(minFVal);
                if (maxFVal > mgd.getMaxVal()) mgd.setMaxVal(maxFVal);

				// Mask
				cv::Mat maskroi (mask, rects[f][r]);
                cv::Mat indexedmaskroi;
                maskroi.copyTo(indexedmaskroi, maskroi == (m_MasksOffset + r));
				GridMat gmask (indexedmaskroi, hp, wp);
				mgd.addGridMask( gmask );
                
//                cout << "M";
                
                // Mask offset
                mgd.addGridMaskOffset(m_MasksOffset + r);
                
//                cout << "o";
                
                // Scene id
                mgd.addSceneID(mgd.getNumOfScenes());
                
//                cout << "s";
                
				// Frame id
				mgd.addGridFrameID(f);
                
//                cout << "i";
                
                // Frame path
                mgd.addFramePath(m_ScenesPaths[sid]);
                
//                cout << "p";
                
                // Frame filename
                mgd.addFrameFilename(framesFilenames[f]);
                
//                cout << "n";
                
                // Mask filename
                mgd.addMaskFilename(masksFilenames[f]);
                
//                cout << "m";

                // Frame resolution
                mgd.addFrameResolution(frame.cols, frame.rows);
                
//                cout << "r";

                // Bounding rect
				mgd.addGridBoundingRect(rects[f][r]);
                
//                cout << "b";
                
				// Tag
				mgd.addTag(tags[f][r]);
                
//                cout << "t";
                
                // Cells' validness
                cv::Mat validnesses = gmask.findNonZero<unsigned char>();
                mgd.addValidnesses(validnesses);
//                cout << "v";
                
                // Partition idx
                mgd.addElementPartition(partition.at<int>(f,0));
                
//                cout << "p";
			}
        }
	}
    
    mgd.addScenePath(m_ScenesPaths[sid]);
}

/*
 * Read only the metadata (frames' filenames, bounding rects, tags, etc)
 */

void ModalityReader::readAllScenesMetadata(string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    mgd.clear();
    mgd.setModality(modality);
    mgd.setHp(hp);
    mgd.setWp(wp);
    
    for (int s = 0; s < m_ScenesPaths.size(); s++)
    {
        readSceneMetadata(s, modality, filetype, hp, wp, mgd);
    }
}

void ModalityReader::readSceneMetadata(unsigned int sid, string modality, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    if (mgd.getModality().compare("") == 0)
        mgd.setModality(modality);
    else
        assert(mgd.getModality().compare(modality) == 0);
    
    if (mgd.getHp() == 0)
        mgd.setHp(hp);
    else
        assert(mgd.getHp() == hp);
    
    if (mgd.getWp() == 0)
        mgd.setWp(wp);
    else
        assert(mgd.getWp() == wp);
    
	// auxiliary
	vector<string> framesFilenames; // Frames' filenames from <dataDir>/Frames/<modality>/
    vector<string> masksFilenames;
	vector<vector<cv::Rect> > rects; // Bounding rects at frame level (having several per frame)
	vector<vector<int> > tags; // Tags corresponding to the bounding rects
    
    if (modality.compare("Motion") == 0)
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Frames/Color/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/Color.yml", rects, tags);
    }
    else if (modality.compare("Ramanan") == 0)
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Maps/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/" + modality + ".yml", rects, tags);
    }
    else
    {
        loadFilenames	 (m_ScenesPaths[sid] + "Frames/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (m_ScenesPaths[sid] + "Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(m_ScenesPaths[sid] + "Masks/" + modality + ".yml", rects, tags);
    }
    
    assert (framesFilenames.size() == masksFilenames.size());
    
    cv::Mat partition = getScenePartition(sid);

    // Load frame-wise (Mat), extract the roi represented by the bounding boxes,
    // grid the rois (GridMat), and store in GridModalityData object
    
    mgd.setHp(hp);
    mgd.setWp(wp);
    
	for (int f = 0; f < framesFilenames.size(); f++)
	{
        if (rects[f].size() < 1)
            continue;
        
        // Load the frame and its mask
        string maskPath;
        
        if (modality.compare("Motion") == 0)
            maskPath = m_ScenesPaths[sid] + "Masks/Color/" + masksFilenames[f] + ".png";
        else
            maskPath = m_ScenesPaths[sid] + "Masks/" + modality + "/" + masksFilenames[f] + ".png";
        
		cv::Mat mask = cv::imread(maskPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        
        // Look the bounding rects in it...
		for (int r = 0; r < rects[f].size(); r++)
		{
            // Create the grid structures
			if (rects[f][r].height >= hp && rects[f][r].width >= wp)
			{                
                // Scene id
                mgd.addSceneID(mgd.getNumOfScenes());
                
				// Frame id
				mgd.addGridFrameID(f);
                
                // Mask offset
                mgd.addGridMaskOffset(m_MasksOffset + r);
                
                // Frame path
                mgd.addFramePath(m_ScenesPaths[sid]);
                
                // Frame filename
                mgd.addFrameFilename(framesFilenames[f]);
                
                // Mask filename
                mgd.addMaskFilename(masksFilenames[f]);
                
                // Frame resolution
                mgd.addFrameResolution(mask.cols, mask.rows);
                
                // Bounding rect
				mgd.addGridBoundingRect(rects[f][r]);
                
				// Tag
				mgd.addTag(tags[f][r]);
                
                // Cells' validness
				cv::Mat maskroi (mask, rects[f][r]);
                cv::Mat indexedmaskroi;
                maskroi.copyTo(indexedmaskroi, maskroi == (m_MasksOffset + r));
				GridMat gmask (indexedmaskroi, hp, wp);
                cv::Mat validnesses = gmask.findNonZero<unsigned char>();
                mgd.addValidnesses(validnesses);
                
                // Partition idx
                mgd.addElementPartition(partition.at<int>(f,0));
			}
		}
	}
    
    mgd.addScenePath(m_ScenesPaths[sid]);
}

void ModalityReader::overlapreadScene(string predictionType, string modality, string dataPath, string scenePath, const char *filetype, ModalityData &md)
{
 
    vector<cv::Mat> predictions;
    vector<cv::Mat> bsMasks;
    vector<cv::Mat> gtMasks;
    
    vector<string> predictionFilenames;
    vector<string> bsMasksFilenames;
    vector<string> gtMasksFilenames;
    
    loadDataToMats   (dataPath + scenePath + "Maps/" +  predictionType + "/Predictions/", ".png", predictions, predictionFilenames);
    loadDataToMats   (dataPath +  scenePath + "Masks/" + modality + "/", ".png", bsMasks, bsMasksFilenames);
    loadDataToMats   (dataPath + scenePath + "GroundTruth/" + modality + "/", ".png", gtMasks, gtMasksFilenames);
    
    assert(bsMasks.size() == gtMasks.size() && bsMasksFilenames.size() == gtMasksFilenames.size());
    
    vector<cv::Mat> predictionMasks;
    vector<cv::Mat> groundTruthMasks;
    
    int predictionsIndex = 0;
    for(int i = 0; i < bsMasksFilenames.size(); i++)
    {
        
        if(gtMasks[i].channels() > 1)
        {
            vector<cv::Mat> auxGt;
            split(gtMasks[i], auxGt);
            groundTruthMasks.push_back(auxGt[0]);
        } else {
            groundTruthMasks.push_back(gtMasks[i]);
        }
        
        //cout << bsMasksFilenames[i] << endl; //debug
        if(bsMasksFilenames[i].compare(predictionFilenames[predictionsIndex]) == 0)
        {
            //cout << predictionFilenames[predictionsIndex] << endl; //debug
            
            //threshold(predictions[predictionsIndex],predictions[predictionsIndex],1,255,CV_THRESH_BINARY);
            /*
           vector<int> un;
            findUniqueValues(predictions[predictionsIndex], un);
            cout << "Before1: ";
            for(int a = 0; a < un.size(); a++)
            {
                cout << un[a] << " ";
            }
            cout << endl;
            
            vector<int> tres;
            findUniqueValues(bsMasks[i], tres);
            cout << "Before2: ";
            for(int a = 0; a < tres.size(); a++)
            {
                cout << tres[a] << " ";
            }
            cout << endl;*/
            
            cv::Mat auxMask;
            bsMasks[i].copyTo(auxMask, predictions[predictionsIndex]);
            predictionMasks.push_back(auxMask);
            
           /* vector<int> dos;
            findUniqueValues(auxMask, dos);
            cout << "After: ";
            for(int a = 0; a < dos.size(); a++)
            {
                cout << dos[a] << " ";
            }
            cout << endl;*/
            
            predictionsIndex++;
        }
        else
        {
            predictionMasks.push_back(cv::Mat::zeros(bsMasks[i].rows, bsMasks[i].cols, CV_8UC1));
        }
    }
    
    //debug
    /*
    for(int i = 0; i < predictionMasks.size(); i++)
    {
        imshow("predictedMasks", predictionMasks[i]);
        imshow("gtMasks", groundTruthMasks[i]);
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    */
    
    md.setPredictedMasks(predictionMasks);
    md.setGroundTruthMasks(groundTruthMasks);
    
    predictions.clear();
    bsMasks.clear();
    gtMasks.clear();
    predictionMasks.clear();
    groundTruthMasks.clear();
    predictionFilenames.clear();
    bsMasksFilenames.clear();
    gtMasksFilenames.clear();
}

void ModalityReader::overlapreadScene(string predictionType, string modality, string scenePath, const char* filetype, ModalityData& md)
{
    overlapreadScene(predictionType, modality, m_DataPath, scenePath, filetype, md);
}


/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void ModalityReader::loadFilenames(string dir, const char* filetype, vector<string>& filenames)
{
	filenames.clear();

    const char* path = dir.c_str();
    string extension = "." + string(filetype);

	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(extension) == 0) )
			{
				string filename = iter->path().filename().string();
				filenames.push_back(filename.substr(0,filename.size()-4));
            }
		}
	}
}

/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void ModalityReader::loadDataToMats(string dir, const char* filetype, vector<cv::Mat> & frames)
{
    const char* path = dir.c_str();
    string extension = "." + string(filetype);

	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(extension) == 0) )
			{
                //cout << iter->path().string() << endl; //debug
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
void ModalityReader::loadDataToMats(string dir, const char* filetype, vector<cv::Mat> & frames, vector<string>& indices)
{
    const char* path = dir.c_str();
    string extension = "." + string(filetype);
    
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(extension) == 0) )
			{
                //cout << iter->path().string() << endl; //debug
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
            
            frame_rects.push_back( cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1) );
        }

        rects.push_back(frame_rects);
        tags.push_back(w);
    }
    
    fs.release();
}

/*
 * Load the people data (bounding boxes coordinates)
 */
void ModalityReader::loadBoundingRects(string file, vector< vector<cv::Rect> > & rects)
{
    cv::FileStorage fs;
    fs.open(file.c_str(), cv::FileStorage::READ);
    
    int num_frames;
    fs["num_frames"] >> num_frames;
    
    for (int i = 0; i < num_frames; i++)
    {
        stringstream ss;
        ss << i;
        
        vector<int> v;
        fs[string("coords_") + ss.str()] >> v;
        
        vector<cv::Rect> frame_rects;
        for (int j = 0; j < v.size() / 4; j++)
        {
            int x0 = v[j*4];
            int y0 = v[j*4+1];
            int x1 = v[j*4+2];
            int y1 = v[j*4+3];
            
            frame_rects.push_back( cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1) );
        }
        
        rects.push_back(frame_rects);
    }
    
    fs.release();
}


void ModalityReader::loadCalibVarsDir(string dir, vector<string>& calibVarsDirs) {
    
    calibVarsDirs.push_back(dir);
    
}

/*
void ModalityReader::agreement(vector<ModalityGridData*> mgds)
{
    // at least one modality is passed
    if (mgds.size() < 1)
        cerr << "[ModalityReader::agree] needs at least one modality" << endl;
    
    // get the first modality's parameters
    int hp0 = mgds[0]->getHp();
    int wp0 = mgds[0]->getWp();
    int ntags0 = mgds[0]->getTags().size();
    // ...
    
    // check if the rest agree with the 0-th modality
    for (int i = 1; i < mgds.size(); i++)
    {
        if (mgds[i]->getHp() != hp0 || mgds[i]->getWp() != wp0)
        {
            cerr << "[ModalityReader::agree] grid dimensions do not coincide" << endl;
            return;
        }
        
        if (mgds[i]->getTags().size() != ntags0)
        {
            cerr << "[ModalityReader::agree] number of tags do not coincide" << endl;
            return;
        }
    }
    
    // perform the validnesses agreement
    // (this can be used to have the same number of cell descriptors through all modalities)
//    for (int i = 0; i < ntags0; i++)
//    {
//        cv::Mat validness = cv::Mat::ones(hp0, wp0, cv::DataType<unsigned char>::type);
//        for (int m = 0; m < mgds.size(); m++)
//        {
//            cv::bitwise_and(validness, mgds[m]->getValidnesses(i), validness);
//        }
//        
//        for (int m = 0; m < mgds.size(); m++)
//        {
//            mgds[m]->setValidnesses(i, validness);
//        }
//    }
    for (int i = 0; i < hp0; i++) for (int j = 0; j < wp0; j++)
    {
        vector<int> counts(mgds.size(), 0);
        
        vector<cv::Mat> descriptors(mgds.size());
        
        for (int k = 0; k < ntags0; k++)
        {
            bool allValids = true;
            for (int m = 0; m < mgds.size() && allValids; m++)
                allValids = mgds[m]->getValidnesses(k).at<unsigned char>(i,j);
            
            if (allValids)
            {
                for (int m = 0; m < mgds.size(); m++)
                    descriptors[m].push_back(mgds[m]->getDescriptor(i, j, counts[m]++));
            }
            else
            {
                for (int m = 0; m < mgds.size(); m++)
                    if (mgds[m]->getValidnesses(k).at<unsigned char>(i,j))
                    {
                        mgds[m]->setValidness(false, i, j, k);
                        counts[m]++;
                    }
            }
        }
        
        for (int m = 0; m < mgds.size(); m++)
            mgds[m]->setDescriptors(descriptors[m], i, j);
    }
}
*/

void ModalityReader::getBoundingBoxesInMask(cv::Mat mask, vector<cv::Rect>& boxes)
{
    boxes.clear();
    
    cvtColor(mask, mask, CV_RGB2GRAY);
    
    for (int l = 0; l < ((int) m_MaxOffset); l++)
    {
        cv::Mat binaryMask = (mask == m_MasksOffset + l);
        
        vector<vector<cv::Point> > contours;
        findContours(binaryMask,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //find contours in groundtruth
        vector<vector<cv::Point> > contours_poly(contours.size());
        vector<cv::Rect> boundRect(contours.size());
        
        for (unsigned int i = 0; i< contours.size(); i++)
        {
            //find bounding boxes around ground truth contours
            approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 2, true );
            boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
            boxes.push_back(boundRect[i]);
        }
    }
}

void ModalityReader::loadDescription(string filePath, ModalityGridData& mgd)
{
    mgd.loadDescription(m_ScenesPaths, filePath);
}

void ModalityReader::getBoundingBoxesFromGroundtruthMasks(string modality, vector<string> sceneDirs, vector<vector<cv::Rect> >& boxes)
{
    boxes.clear();
    
    for (int s = 0; s < sceneDirs.size(); s++)
    {
        getBoundingBoxesFromGroundtruthMasks(modality, sceneDirs[s], boxes);
    }
}

void ModalityReader::getBoundingBoxesFromGroundtruthMasks(string modality, string sceneDir, vector<vector<cv::Rect> >& boxes)
{
    string dir = m_DataPath + sceneDir + "GroundTruth/" + modality;
    vector<string> filenames;
    loadFilenames(dir, "png", filenames);
    for (int f = 0; f < filenames.size(); f++)
    {
        string filePath = dir + "/" + filenames[f] + ".png";
        cv::Mat mask = cv::imread(filePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        
        vector<cv::Rect> maskBoxes;
        getBoundingBoxesInMask(mask, maskBoxes);
        
        boxes.push_back(maskBoxes);
    }
}
