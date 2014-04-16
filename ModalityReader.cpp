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
			if ( is_directory( *iter ) && iter->path().string().find("Scene") != string::npos)
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
        loadDataToMats   (m_ScenesPaths[i] + "/Frames/" + modality + "/", "jpg", frames, filenames);
        loadDataToMats   (m_ScenesPaths[i] + "/Masks/" + modality + "/", "png", masks);
        loadDataToMats(m_ScenesPaths[i] + "/GroundTruth/" + modality + "/", "png", gtMasks);
        loadBoundingRects(m_ScenesPaths[i] + "/Masks/" + modality + ".yml", rects, tags);
        if(modality.compare("Thermal") == 0) {
            loadCalibVarsDir (m_ScenesPaths[i] + "/calibVars.yml", calibVars);
        }
        if(modality.compare("Depth") == 0) {
            loadDataToMats(m_ScenesPaths[i] + "/Frames/" + modality + "Raw/", "png", regFrames);
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

void ModalityReader::read(string modality, string sceneDir, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    vector<string> scenesDirs;
    scenesDirs.push_back(sceneDir);
    
    read(modality, m_DataPath, scenesDirs, filetype, hp, wp, mgd);
}

void ModalityReader::read(string modality, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    read(modality, m_DataPath, sceneDirs, filetype, hp, wp, mgd);
}

void ModalityReader::read(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    mgd.clear();
    mgd.setModality(modality);
    mgd.setHp(hp);
    mgd.setWp(wp);
    
    for (int s = 0; s < sceneDirs.size(); s++)
    {
        readScene(modality, dataPath + sceneDirs[s], filetype, hp, wp, mgd);
    }
}

void ModalityReader::readScene(string modality, string scenePath, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
	// auxiliary
	vector<string> framesFilenames; // Frames' filenames from <dataDir>/Frames/<modality>/
    vector<string> masksFilenames; // Corresponding masks' filenames
	vector<vector<cv::Rect> > rects; // Bounding rects at frame level (having several per frame)
	vector<vector<int> > tags; // Tags corresponding to the bounding rects
    
    // Motion & Ramanan modalities need special treatment for reading
    // .. In motion case, the optical flow vectors must be computed for every pair of frames
    // .. In ramanan, the frames are already computed probability maps (from Matlab)
    if (modality.compare("Motion") == 0)
    {
        loadFilenames	 (scenePath + "/Frames/Color/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/Color.yml", rects, tags);
    }
    else if (modality.compare("Ramanan") == 0)
    {
        loadFilenames	 (scenePath + "/Maps/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/Color.yml", rects, tags);
    }
    else
    {
        loadFilenames	 (scenePath + "/Frames/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/" + modality + ".yml", rects, tags);
    }

    assert (framesFilenames.size() == masksFilenames.size());
    
    // Load frame-wise (Mat), extract the roi represented by the bounding boxes,
    // grid the rois (GridMat), and store in GridModalityData object
    
	for (int f = 0; f < framesFilenames.size(); f++)
	{
        if (rects[f].size() < 1)
            continue;
        
        // Load the frame and its mask
        string framePath, maskPath;
        
        if (modality.compare("Motion") == 0)
        {
            framePath = scenePath + "/Frames/Color/" + framesFilenames[f] + "." + filetype;
            maskPath = scenePath + "/Masks/Color/" + masksFilenames[f] + ".png";
        }
        if (modality.compare("Ramanan") == 0)
        {
            framePath = scenePath + "/Maps/Ramanan/" + framesFilenames[f] + "." + filetype;
            maskPath = scenePath + "/Masks/Color/" + masksFilenames[f] + ".png";
        }
        else
        {
            framePath = scenePath + "/Frames/" + modality + "/" + framesFilenames[f] + "." + filetype;
            maskPath = scenePath + "/Masks/" + modality + "/" + masksFilenames[f] + ".png";
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
            cv::Mat prevFrame;
            cv::Mat currFrame(frame);
            
            if (f == 0)
                currFrame.copyTo(prevFrame);
            else
                prevFrame = cv::imread(scenePath + "/Frames/Color/" + framesFilenames[f-1] + "." + filetype,
                                       CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            
            MotionFeatureExtractor::computeOpticalFlow(pair<cv::Mat,cv::Mat>(prevFrame,currFrame), frame);
        }
        // --------------------------------------------------------------------------------------
        
        // Look the bounding rects in it...

		for (int r = 0; r < rects[f].size(); r++)
		{
            // Create the grid structures
            
			if (rects[f][r].height >= hp && rects[f][r].width >= wp)
			{
				cv::Mat subjectroi (frame, rects[f][r]); // Get a roi in frame defined by the rectangle.
				GridMat gsubject (subjectroi, hp, wp);
				mgd.addGridFrame( gsubject );
                
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
                
                // Mask offset
                mgd.addGridMaskOffset(m_MasksOffset + r);
                
                // Scene id
                mgd.addSceneID(mgd.getNumOfScenes());
                
				// Frame id
				mgd.addGridFrameID(f);
                
                // Frame path
                mgd.addFramePath(scenePath);
                
                // Frame filename
                mgd.addFrameFilename(framesFilenames[f]);
                
                // Mask filename
                mgd.addMaskFilename(masksFilenames[f]);

                // Frame resolution
                mgd.addFrameResolution(frame.cols, frame.rows);

                // Bounding rect
				mgd.addGridBoundingRect(rects[f][r]);
                
				// Tag
				mgd.addTag(tags[f][r]);
                
                // Cells' validness
                cv::Mat validnesses = gmask.findNonZero<unsigned char>();
                mgd.addValidnesses(validnesses);
			}
		}
	}
    
    mgd.addScenePath(scenePath);
}

/*
 * Read only the metadata (frames' filenames, bounding rects, tags, etc)
 */
void ModalityReader::mockread(string modality, string sceneDir, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    vector<string> scenesDirs;
    scenesDirs.push_back(sceneDir);
    
    mockread(modality, m_DataPath, scenesDirs, filetype, hp, wp, mgd);
}

void ModalityReader::mockread(string modality, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    mockread(modality, m_DataPath, sceneDirs, filetype, hp, wp, mgd);
}

void ModalityReader::mockread(string modality, string dataPath, vector<string> sceneDirs, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
    mgd.clear();
    mgd.setModality(modality);
    mgd.setHp(hp);
    mgd.setWp(wp);
    
    for (int s = 0; s < sceneDirs.size(); s++)
    {
        mockreadScene(modality, dataPath + sceneDirs[s], filetype, hp, wp, mgd);
    }
}

void ModalityReader::mockreadScene(string modality, string scenePath, const char* filetype, int hp, int wp, ModalityGridData& mgd)
{
	// auxiliary
	vector<string> framesFilenames; // Frames' filenames from <dataDir>/Frames/<modality>/
    vector<string> masksFilenames;
	vector<vector<cv::Rect> > rects; // Bounding rects at frame level (having several per frame)
	vector<vector<int> > tags; // Tags corresponding to the bounding rects
    
    if (modality.compare("Motion") == 0)
    {
        loadFilenames	 (scenePath + "/Frames/Color/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/Color/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/Color.yml", rects, tags);
    }
    else if (modality.compare("Ramanan") == 0)
    {
        loadFilenames	 (scenePath + "/Maps/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/" + modality + ".yml", rects, tags);
    }
    else
    {
        loadFilenames	 (scenePath + "/Frames/" + modality + "/", filetype, framesFilenames);
        loadFilenames	 (scenePath + "/Masks/" + modality + "/", "png", masksFilenames);
        
        loadBoundingRects(scenePath + "/Masks/" + modality + ".yml", rects, tags);
    }
    
    cv::Mat_<int> counts (100, 1);
    counts.setTo(0);
    for (int i = 0; i < rects.size(); i++)
    {
        counts.at<int>(rects[i].size(),0)++;
    }
    cout << counts << endl;
    
    assert (framesFilenames.size() == masksFilenames.size());
    
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
            maskPath = scenePath + "/Masks/Color/" + masksFilenames[f] + ".png";
        else
            maskPath = scenePath + "/Masks/" + modality + "/" + masksFilenames[f] + ".png";
        
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
                mgd.addFramePath(scenePath);
                
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
			}
		}
	}
    
    mgd.addScenePath(scenePath);
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
    
    vector<cv::Mat> predictionMasks;
    
    int predictionsIndex = 0;
    for(int i = 0; i < bsMasksFilenames.size(); i++)
    {
        cout << bsMasksFilenames[i] << endl; //debug
        if(bsMasksFilenames[i].compare(predictionFilenames[predictionsIndex]) == 0)
        {
            cout << predictionFilenames[predictionsIndex] << endl; //debug
            predictionMasks.push_back(bsMasks[i].mul(predictions[predictionsIndex]));
            predictionsIndex++;
        }
        else
        {
            predictionMasks.push_back(cv::Mat::zeros(bsMasks[i].rows, bsMasks[i].cols, CV_8UC1));
        }
    }
    
    //debug
    for(int i = 0; i < predictionMasks.size(); i++)
    {
        imshow("predictedMasks", predictionMasks[i]);
        cv::waitKey(10);
    }
    
    
    //TODO: change this! - or erase it -
    //vector<cv::Mat> masks(2); //1. bsmasks, 2.predictions
    //split(md.getPredictedMasks(), masks);
    
    //vector<cv::Mat> output;
    
    
    //int predictionIndex = 0;
    //for(int i = 0; i < bsMasks.size(); i++)
    //{
    //cv::Mat aux;
    //cv::Mat mask(predictions[0].rows, predictions[0].cols, CV_8UC2);
    //if(cv::countNonZero(predictions[predictionIndex]) > 0)
    //{
    //    aux = predictions[predictionIndex];
    //    predictionIndex++;
    //}
    //else
    //{
    //    aux = cv::Mat::zeros(predictions[0].rows, predictions[0].cols, CV_8UC1);
    //}
    
    //cv::Mat in[] = {bsMasks[i], aux};
    //int from_to[] = {0,0, 1,1};
    //cv::mixChannels(in, 2, &mask, 1, from_to, 2);
    
    //predictedMasks.push_back(mask);
    //}
    
    
    md.setPredictedMasks(predictionMasks);
    md.setGroundTruthMasks(gtMasks);
    
    predictions.clear();
    bsMasks.clear();
    gtMasks.clear();
    predictionMasks.clear();
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

/**
 * Load data to opencv's cv::Mats
 *
 * This method uses OpenCV and Boost.
 */
void ModalityReader::loadDataToMats(string dir, const char* filetype, vector<cv::Mat> & frames)
{
    const char* path = dir.c_str();
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(filetype) == 0) )
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
	if( exists( path ) )
	{
        boost::filesystem::
		directory_iterator end;
		directory_iterator iter(path);
		for( ; iter != end ; ++iter )
		{
			if ( !is_directory( *iter ) && (iter->path().extension().string().compare(filetype) == 0) )
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