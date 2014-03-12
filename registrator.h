#ifndef REGISTRATOR_H
#define REGISTRATOR_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <dirent.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define MAP_FROM_RGB 0
#define MAP_FROM_THERMAL 1
#define MAP_FROM_DEPTH 2

class Registrator
{
public:
	Registrator();
    
	/* Functions for the pixel-to-pixel registration of tri-modal registration. All registration functions accepts a vector of points as input
     and might therefore be called with multiple points at once
     
     The vector of points is created as:
     std::vector<cv::Point2f> vecRgbCoord;
     vecRgbCoord.push_back(Point2f(x,y));
     
     */
    
	/* computeCorrespondingThermalPointFromRgb handles the registration of RGB points into the corresponding thermal point.
     In order to provide the registration, the function takes as input the corresponding depth coordinate, which might be computed
     from the function computeCorrespondingDepthPointFromRgb */
	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord);
    
	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord,
												 std::vector<int> &bestHom);
    
	/* Full overload of computeCorrespondingThermalPointFromRgb containing full information of the internal functions for debugging purposes */
	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord,
                                                 std::vector<int> vecDepthInMm, std::vector<double>& minDist, std::vector<int> &bestHom,
                                                 std::vector<std::vector<int> > &octantIndices, std::vector<std::vector<double> > &octantDistances,
                                                 std::vector<cv::Point3f> &worldCoordPointvector);
    
	/* computeCorrespondingDepthPointFromRgb handles the registration of RGB points into the corresponding point in the depth image
     As the function uses a look-up-table for the registration, only the RGB point needs to be provided */
	void computeCorrespondingDepthPointFromRgb(std::vector<cv::Point2f> vecRgbCoord,std::vector<cv::Point2f> & vecDCoord);
    
	/* computeCorrespondingDepthPointFromRgb handles the registration of depth points into the corresponding point in the RGB image
     As the function uses a look-up-table for the registration, only the depth point needs to be provided */
	void computeCorrespondingRgbPointFromDepth(std::vector<cv::Point2f> vecDCoord,std::vector<cv::Point2f> & vecRgbCoord);
    
	/* computeCorrespondingRgbPointFromThermal handles the registration of thermal points (in the thermal modality). The registration is handled without providing any
     depth coordinate, as the thermal camera does not contain any direct connection to the depth information of the Kinect camera*/
	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord);
    
	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord, std::vector<double>& minDist,
												 std::vector<int> &bestHom, std::vector<std::vector<int> > &octantIndices, std::vector<std::vector<double> > &octantDistances);
    
	/* Full overload of computeCorrespondingRgbPointFromThermal containing full information of the internal functions for debugging purposes */
	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord, std::vector<double>& minDist,
												 std::vector<int> &bestHom, std::vector<std::vector<int> > &octantIndices, std::vector<std::vector<double> > &octantDistances,
												 std::vector<cv::Point3f> &worldCoordPointstdvector);
    
	// Utility functions for loading and registering RGB and depth images
	void loadTransformRgb();
	void loadTransformDepth();
    
	// Utility functions for loading and registering contours of multimodal imagery
	void loadRegSaveContours();
	void loadRegSaveContours(const char* pathMasksColor, const char* pathMasksDepth,
                             const char* pathMasksThermal, const char* pathDepth);
    void loadRegSaveContours(std::vector<cv::Mat> masksColor, std::vector<cv::Mat> masksDepth,
                             std::vector<cv::Mat> & masksThermal, std::vector<cv::Mat> depth);
    
	void loadMinCalibrationVars(std::string calFile);
    
	void showModalityImages(std::string rgbPath,std::string tPath,std::string dPath,int imgNbr);
	void initWindows(int imgNbr);
    
	// Mouse handling of OpenCV HighGUI
	void rgbOnMouse( int event, int x, int y, int flags);
	void thermalOnMouse( int event, int x, int y, int flags);
	void depthOnMouse( int event, int x, int y, int flags);
    
	std::string getRgbImgPath();
	std::string getTImgPath();
	std::string getDImgPath();
	void setRgbImgPath(std::string path);
	void setTImgPath(std::string path);
	void setDImgPath(std::string path);
    
	// If the provided images or coordinates are undistorted, this variable should be true. Otherwise, false
	void toggleUndistortion(bool undistort);
    
	// Use for debugging and optimization purposes
	void setDiscardedHomographies(std::vector<int> discardedHomographies);
	void setUsePrevDepthPoint(bool value=true);
    
private:
    
	/* computeHomographyMapping handles the mapping of RGB <-> Thermal.
     It is called inside computeCorrespondingRgbPointFromThermal and computeCorrespondingThermalPointFromRgb */
	void computeHomographyMapping(std::vector<cv::Point2f>& vecUndistRgbCoord, std::vector<cv::Point2f>& vecUndistTCoord, std::vector<cv::Point2f> vecDCoord,
                                  std::vector<int> vecDepthInMm, std::vector<double>& minDist, std::vector<int> &bestHom, std::vector<std::vector<int> > &octantIndices,
                                  std::vector<std::vector<double> > &octantDistances, std::vector<cv::Point3f> &worldCoordPointvector);
    
	/*	TrilinearHomographyInterpolator finds the nearest point for each quadrant in 3D space and calculates weights
     based on trilinear interpolation for the input 3D point. The function returns a list of weights of the points
     used for the interpolation */
	void trilinearInterpolator(cv::Point3f inputPoint, std::vector<cv::Point3f> &sourcePoints, std::vector<double> &precomputedDistance,
							   std::vector<double> &weights, std::vector<int> &nearestSrcPointInd, std::vector<double> &nearestSrcPointDist);
    
	/* weightedHomographyMapper maps the undistPoint based by a weighted sum of the provided homographies weighted by homWeights */
	void weightedHomographyMapper(std::vector<cv::Point2f> undistPoint, std::vector<cv::Point2f> &estimatedPoint, std::vector<cv::Mat> &homographies,
								  std::vector<double> &homWeights);
	
	/* MyDistortPoints distort points, used if the source image is undistorted*/
	void MyDistortPoints(const std::vector<cv::Point2f> src, std::vector<cv::Point2f> & dst,
                         const cv::Mat & cameraMatrix, const cv::Mat &distorsionMatrix);
    
	float backProjectPoint(float point, float focalLength, float principalPoint, float zCoord);
    
	float forwardProjectPoint(float point, float focalLength, float principalPoint, float zCoord);
    
	// lookUpDepth gets the current depth of the point in the depth image
	float lookUpDepth(cv::Mat depthImg, cv::Point2f dCoord, bool SCALE_TO_THEORETICAL);
    
	// Helper functions for the registration of contours:
	void getRegisteredContours(std::vector<cv::Point> contour, std::vector<cv::Point> erodedContour, std::vector<cv::Point>& dContour,
							   std::vector<cv::Point>& tContour, cv::Mat depthImg);
    
	void drawRegisteredContours(cv::Mat rgbContourImage, cv::Mat& depthContourImage, cv::Mat& thermalContourImage, cv::Mat depthImg,
								bool preserveColors = false);
    
	void saveRegisteredContours(cv::Mat depthContourImage, cv::Mat thermalContourImage, std::string depthSavePath, std::string thermalSavePath,
								std::string imgNbr);
    
	void depthOutlierRemovalLookup(std::vector<cv::Point2f> vecDCoord, std::vector<int> &vecDepthInMm);
    
	void buildContourDirectory(std::string rgbLoadPath, std::vector<std::string> &rgbContours);
    
	// Helper functions for the registration of RGB and depth images:
	void transformRgbImageToDepth(cv::Mat rgbSourceImg, cv::Mat &dDestImg, std::vector<cv::Point2f> rgbPoints, std::vector<cv::Point2f> dPoints);
    
	void transformDepthImageToRgb(cv::Mat dSourceImg, cv::Mat &rgbDestImg, std::vector<cv::Point2f> rgbPoints, std::vector<cv::Point2f> dPoints);
    
	void buildImageDirectory(std::string rgbLoadPath, std::vector<std::string>& rgbIndices);
    
	void buildImageDirectory(std::string rgbLoadPath, std::vector<std::string>& rgbIndices, std::string filetype);
    
	void saveRegisteredImages(cv::Mat depthImage, std::string depthSavePath, std::string imgNbr);
    
	/* markCorrespondingPoints takes as input a coordinate of the corresponding modality and marks the registered point
     in the remaining modalities*/
	void markCorrespondingPointsRgb(cv::Point2f rgbCoord);
    
	void markCorrespondingPointsThermal(cv::Point2f tCoord);
    
	void markCorrespondingPointsDepth(cv::Point2f dCoord);
    
	void markCorrespondingPoints(cv::Point2f rgbCoord, cv::Point2f tCoord, cv::Point2f dCoord, int homInd, int MAP_TYPE);
    
	static void updateFrames(int imgNbr, void* obj);
    
	struct calibrationParams {
		// Calibration parameters which are loaded from the calibVars.yml-file
        
		// Depth calibration parameters
		float depthCoeffA; // y = a*x+b
		float depthCoeffB;
        
		int WIDTH;
		int HEIGHT;
        
		// Camera calibration parameters
		std::vector<int> activeImages;
		cv::Mat rgbCamMat, tCamMat, rgbDistCoeff, tDistCoeff;
        
		// Rectification matrices and mappings
		std::vector<cv::Mat> planarHom, planarHomInv;
		std::vector<cv::Point3f> homDepthCentersRgb,homDepthCentersT;
		int defaultDepth;
        
		// Depth calibration parameters
		cv::Mat rgbToDCalX,rgbToDCalY,dToRgbCalX,dToRgbCalY;
	} stereoCalibParam;
    
	struct registrationSettings {
		bool UNDISTORT_IMAGES;
		bool USE_PREV_DEPTH_POINT;
        
		int nbrClustersForDepthOutlierRemoval;
		int depthProximityThreshold;
		std::vector<int> discardedHomographies;
	} settings;
    
	cv::Mat rgbImg, tImg, dImg;
	std::string rgbImgPath, tImgPath, dImgPath;
};

#endif // REGISTRATOR_H

// Declare function prototypes outside of the Registrator class
void wrappedRgbOnMouse(int event, int x, int y, int flags, void* ptr);
void wrappedThermalOnMouse(int event, int x, int y, int flags, void* ptr);
void wrappedDepthOnMouse(int event, int x, int y, int flags, void* ptr);
