// registrator.cpp : Defines the entry point for the console application.
//
#include "registrator.h"

using namespace std;
using namespace cv;

Registrator::Registrator()
{
	// Initialize
}

float Registrator::backProjectPoint(float point, float focalLength, float principalPoint, float zCoord)
{
	float projectedPoint;

 	projectedPoint = (point * zCoord - principalPoint * zCoord)/(focalLength);
	return projectedPoint;
}

float Registrator::forwardProjectPoint(float point, float focalLength, float principalPoint, float zCoord)
{
	float projectedPoint;

	projectedPoint = (point * focalLength)/zCoord + principalPoint;
	return projectedPoint;
}

float Registrator::lookUpDepth(Mat depthImg, Point2f dCoord, bool SCALE_TO_THEORETICAL)
{
	float depthInMm;
	// Look up the 'vanilla' depth in the depth image
	depthInMm = float(depthImg.at<unsigned short>(int(dCoord.y-1),int(dCoord.x-1))*0.1);

	if (SCALE_TO_THEORETICAL)
	{
		// Transfer the depth to the "theoretical" depth range as provided by the camera calibration
		depthInMm = stereoCalibParam.depthCoeffA*depthInMm+stereoCalibParam.depthCoeffB;
	}

	return depthInMm;
}

void Registrator::computeCorrespondingRgbPointFromDepth(vector<Point2f> vecDCoord,vector<Point2f> & vecRgbCoord)
{
	vector<Point2f> vecDistRgbCoord,vecUndistRgbCoord,vecRecRgbCoord;
	Point2f tmpPoint;
	Point2f prevPoint;

	// Find the distorted rgb point
	for (size_t i = 0; i < vecDCoord.size(); i++)
	{
		if ((vecDCoord[i].y > 0) && (vecDCoord[i].x > 0) && (vecDCoord[i].y <= stereoCalibParam.HEIGHT) && (vecDCoord[i].x <= stereoCalibParam.WIDTH)) {
			tmpPoint.x = stereoCalibParam.dToRgbCalX.at<short>(int(vecDCoord[i].y-1), int(vecDCoord[i].x-1)); // Look up the corresponding x-point in the depth image
			tmpPoint.y = stereoCalibParam.dToRgbCalY.at<short>(int(vecDCoord[i].y-1), int(vecDCoord[i].x-1)); // Look up the corresponding y-point in the depth image

			if ((tmpPoint.x == 0) || (tmpPoint.y == 480) || (tmpPoint.y == 0))
			{
				// If the point is inside the frame of the depth camera, look up neighbouring points and perform bilinear interpolation
				Point2f interpolPoint(0,0), tmpLookupPoint;;
				int lookupCount = 0, xOffset, yOffset;

				for (int j = 0; j < 4; ++j) {
					
					switch (j) {
					case 0: 
						xOffset = -1; 
						yOffset = -1;
						break;
					case 1: 
						xOffset = +1;
						yOffset = -1;
						break;
					case 2: 
						xOffset = +1;
						yOffset = +1;
						break;
					case 3: 
						xOffset = -1;
						yOffset = +1;
						break;
					}
					
					if (((vecDCoord[i].y+yOffset) > 0) && ((vecDCoord[i].x+xOffset) > 0) && ((vecDCoord[i].y+yOffset) <= stereoCalibParam.HEIGHT) 
						&& ((vecDCoord[i].x+xOffset) <= stereoCalibParam.WIDTH)) {
						tmpLookupPoint.x = stereoCalibParam.dToRgbCalX.at<short>(int(vecDCoord[i].y-1+yOffset), int(vecDCoord[i].x-1+xOffset)); // Look up the corresponding x-point in the depth image
						tmpLookupPoint.y = stereoCalibParam.dToRgbCalY.at<short>(int(vecDCoord[i].y-1+yOffset), int(vecDCoord[i].x-1+xOffset)); // Look up the corresponding y-point in the depth image

						if ((tmpLookupPoint.x != 0) && (tmpLookupPoint.y != 480) && (tmpLookupPoint.y != 0)) {
							interpolPoint.x = interpolPoint.x + tmpLookupPoint.x;
							interpolPoint.y = interpolPoint.y + tmpLookupPoint.y;
							lookupCount++;
						}
					}	
				}

				if (lookupCount > 0) {
					tmpPoint.x = interpolPoint.x/lookupCount;
					tmpPoint.y = interpolPoint.y/lookupCount;
				} else if (this->settings.USE_PREV_DEPTH_POINT)
				{
					tmpPoint = prevPoint;
				} else {
					tmpPoint = Point2f(0,0);
				}
			}
			vecDistRgbCoord.push_back(tmpPoint);
		} else {
			tmpPoint.x = 1; tmpPoint.y = 1;
			vecDistRgbCoord.push_back(tmpPoint);
		}
	}

	if (settings.UNDISTORT_IMAGES) {
		undistortPoints(vecDistRgbCoord,vecUndistRgbCoord,stereoCalibParam.rgbCamMat,stereoCalibParam.rgbDistCoeff,
				Mat::eye(3,3,CV_32F),stereoCalibParam.rgbCamMat);
	}

	if (settings.UNDISTORT_IMAGES) {
		// If images are undistorted, but not rectified, use undistorted coordinates
		vecRgbCoord = vecUndistRgbCoord;
	} else {
		// Otherwise, use the distorted coordinates
		vecRgbCoord = vecDistRgbCoord;
	}

}

void Registrator::computeCorrespondingDepthPointFromRgb(vector<Point2f> vecRgbCoord,vector<Point2f> & vecDCoord)
{
	vector<Point2f> vecDistRgbCoord, vecUndistRgbCoord;
	Point2f tmpPoint;
	Point2f prevPoint(0, 0);

	// Now, we may re-distort the points so they correspond to the original image
	if (settings.UNDISTORT_IMAGES) {
		// If we need to undistort the point, do so:
		MyDistortPoints(vecRgbCoord,vecDistRgbCoord,stereoCalibParam.rgbCamMat,stereoCalibParam.rgbDistCoeff); 
	} else {
		// Otherwise, the point is already distorted
		vecDistRgbCoord = vecRgbCoord;
	}

	
	for (size_t i = 0; i < vecDistRgbCoord.size(); i++) 
	{
		if ((vecDistRgbCoord[i].y > 0) && (vecDistRgbCoord[i].x > 0) && (vecDistRgbCoord[i].y <= stereoCalibParam.HEIGHT) 
				&& (vecDistRgbCoord[i].x <= stereoCalibParam.WIDTH)) {
			tmpPoint.x = stereoCalibParam.rgbToDCalX.at<short>(int(vecDistRgbCoord[i].y-1), int(vecDistRgbCoord[i].x-1)); // Look up the corresponding x-point in the depth image
			tmpPoint.y = stereoCalibParam.rgbToDCalY.at<short>(int(vecDistRgbCoord[i].y-1), int(vecDistRgbCoord[i].x-1)); // Look up the corresponding y-point in the depth image

			if ((tmpPoint.x == 0) || (tmpPoint.y == 480) || (tmpPoint.y == 0)) {
				// If the point is inside the frame of the RGB camera, look up neighbouring points and perform bilinear interpolation
				Point2f interpolPoint(0,0), tmpLookupPoint;;
				int lookupCount = 0, xOffset, yOffset;

				for (int j = 0; j < 4; ++j) {
					
					switch (j) {
					case 0: 
						xOffset = -1; 
						yOffset = -1;
						break;
					case 1: 
						xOffset = +1;
						yOffset = -1;
						break;
					case 2: 
						xOffset = +1;
						yOffset = +1;
						break;
					case 3: 
						xOffset = -1;
						yOffset = +1;
						break;
					}
					
					if (((vecDistRgbCoord[i].y+yOffset) > 0) && ((vecDistRgbCoord[i].x+xOffset) > 0) && ((vecDistRgbCoord[i].y+yOffset) <= stereoCalibParam.HEIGHT) 
						&& ((vecDistRgbCoord[i].x+xOffset) <= stereoCalibParam.WIDTH)) {
						tmpLookupPoint.x = stereoCalibParam.rgbToDCalX.at<short>(int(vecDistRgbCoord[i].y-1+yOffset), int(vecDistRgbCoord[i].x-1+xOffset)); // Look up the corresponding x-point in the depth image
						tmpLookupPoint.y = stereoCalibParam.rgbToDCalY.at<short>(int(vecDistRgbCoord[i].y-1+yOffset), int(vecDistRgbCoord[i].x-1+xOffset)); // Look up the corresponding y-point in the depth image

						if ((tmpLookupPoint.x != 0) && (tmpLookupPoint.y != 480) && (tmpLookupPoint.y != 0)) {
							interpolPoint.x = interpolPoint.x + tmpLookupPoint.x;
							interpolPoint.y = interpolPoint.y + tmpLookupPoint.y;
							lookupCount++;
						}
					}	
				}

				if (lookupCount > 0) {
					tmpPoint.x = interpolPoint.x/lookupCount;
					tmpPoint.y = interpolPoint.y/lookupCount;
				} else if (this->settings.USE_PREV_DEPTH_POINT)
				{
					tmpPoint = prevPoint;
				} else {
					tmpPoint = Point2f(0,0);
				}
			}
			vecDCoord.push_back(tmpPoint);
		} else {
			

			if (this->settings.USE_PREV_DEPTH_POINT)
			{
				// If the RGB coordinates are not within the camera frame, return the previous value
				vecDCoord.push_back(prevPoint);
			} else {
				vecDCoord.push_back(Point2f(0,0));
			}
		}

		prevPoint = tmpPoint;
	}

}

void Registrator::computeCorrespondingThermalPointFromRgb(vector<Point2f> vecRgbCoord, vector<Point2f>& vecTCoord, vector<Point2f> vecDCoord)
{
	vector<int> vecDepthInMm, bestHom;
	vector<double> minDist;
	vecDepthInMm.resize(0);
	vector<vector<double>> octantDistances;
	vector<vector<int>> octantIndices;
	vector<Point3f> worldCoordPointVector;
	computeCorrespondingThermalPointFromRgb(vecRgbCoord, vecTCoord, vecDCoord, vecDepthInMm, minDist, 
											bestHom, octantIndices, octantDistances, worldCoordPointVector);

}

void Registrator::computeCorrespondingThermalPointFromRgb(vector<Point2f> vecRgbCoord, vector<Point2f>& vecTCoord, vector<Point2f> vecDCoord, 
														  vector<int> &bestHom)
{
	vector<int> vecDepthInMm;
	vector<double> minDist;
	vecDepthInMm.resize(0);
	vector<vector<double>> octantDistances;
	vector<vector<int>> octantIndices;
	vector<Point3f> worldCoordPointVector;
	computeCorrespondingThermalPointFromRgb(vecRgbCoord, vecTCoord, vecDCoord, vecDepthInMm, minDist, 
											bestHom, octantIndices, octantDistances, worldCoordPointVector);
}

void Registrator::computeCorrespondingThermalPointFromRgb(vector<Point2f> vecRgbCoord, vector<Point2f>& vecTCoord, vector<Point2f> vecDCoord, 
											vector<int> vecDepthInMm, vector<double>& minDist, vector<int> &bestHom, vector<vector<int>> &octantIndices, 
											vector<vector<double>> &octantDistances, vector<Point3f> &worldCoordPointVector)
{
	vector<Point2f> vecUndistRgbCoord,vecRecRgbCoord,vecDistRgbCoord,vecRecTCoord, vecUndistTCoord;
	Point2f tmpPoint;
	minDist.clear(); bestHom.clear();

	if (!settings.UNDISTORT_IMAGES) // If the images are not undistorted, produce undistorted coordinates
	{
		undistortPoints(vecRgbCoord,vecUndistRgbCoord, stereoCalibParam.rgbCamMat, stereoCalibParam.rgbDistCoeff,
						Mat::eye(3,3,CV_32F),stereoCalibParam.rgbCamMat);
		vecDistRgbCoord = vecRgbCoord;
	} else {
		vecUndistRgbCoord = vecRgbCoord;
	}

	computeHomographyMapping(vecUndistRgbCoord, vecUndistTCoord, vecDCoord, vecDepthInMm, minDist, bestHom, 
			octantIndices, octantDistances, worldCoordPointVector);

	// If needed, distort the coordinates
	if (!settings.UNDISTORT_IMAGES) // If the images are distorted, produce distorted coordinates
	{
		MyDistortPoints(vecUndistTCoord,vecTCoord,stereoCalibParam.tCamMat,stereoCalibParam.tDistCoeff);
	} else { 
		vecTCoord = vecUndistTCoord;
	}

}

void Registrator::computeCorrespondingRgbPointFromThermal(vector<Point2f> vecTCoord, vector<Point2f>& vecRgbCoord)
{
	vector<double> minDist;
	vector<int> bestHom;
	vector<vector<int>> octantIndices;
	vector<vector<double>> octantDistances;
	vector<Point3f> worldCoordPointVector;

	computeCorrespondingRgbPointFromThermal(vecTCoord, vecRgbCoord, minDist, bestHom, octantIndices, octantDistances, 
											worldCoordPointVector);

	
}

void Registrator::computeCorrespondingRgbPointFromThermal(vector<Point2f> vecTCoord, vector<Point2f>& vecRgbCoord, vector<double>& minDist, 
											 vector<int> &bestHom, vector<vector<int>> &octantIndices, vector<vector<double>> &octantDistances)
{
	vector<Point3f> worldCoordPointVector;
	computeCorrespondingRgbPointFromThermal(vecTCoord, vecRgbCoord, minDist, bestHom, octantIndices, octantDistances, 
											worldCoordPointVector);
}

void Registrator::computeCorrespondingRgbPointFromThermal(vector<Point2f> vecTCoord, vector<Point2f>& vecRgbCoord, vector<double>& minDist, 
											 vector<int> &bestHom, vector<vector<int>> &octantIndices, vector<vector<double>> &octantDistances, 
											 vector<Point3f> &worldCoordPointVector)
{
	vector<Point2f> vecUndistTCoord,vecRecTCoord,vecDistTCoord,vecRecRgbCoord, vecUndistRgbCoord,vecDCoord;
	Point2f tmpPoint;
	minDist.clear(); bestHom.clear();

	if (!settings.UNDISTORT_IMAGES) // If the images are not undistorted, produce undistorted coordinates
	{
		undistortPoints(vecTCoord,vecUndistTCoord, stereoCalibParam.tCamMat, stereoCalibParam.tDistCoeff,
						Mat::eye(3,3,CV_32F),stereoCalibParam.tCamMat);
		vecDistTCoord = vecTCoord;
	} else {
		vecUndistTCoord = vecTCoord;
	}

	vector<int> vecDepthInMm;
	computeHomographyMapping(vecUndistRgbCoord, vecUndistTCoord, vecDCoord, vecDepthInMm, minDist, bestHom,
				octantIndices, octantDistances, worldCoordPointVector);

	// If needed, distort the coordinates
	if (!settings.UNDISTORT_IMAGES) // If the images are distorted, produce distorted coordinates
	{
		MyDistortPoints(vecUndistRgbCoord,vecRgbCoord,stereoCalibParam.rgbCamMat,stereoCalibParam.rgbDistCoeff);
	} else {
		vecRgbCoord = vecUndistRgbCoord;
	}

}

void Registrator::computeHomographyMapping(vector<Point2f>& vecUndistRgbCoord, vector<Point2f>& vecUndistTCoord, vector<Point2f> vecDCoord,
		vector<int> vecDepthInMm, vector<double>& minDist, vector<int> &bestHom, vector<vector<int>> &octantIndices,
		vector<vector<double>> &octantDistances, vector<Point3f> &worldCoordPointVector)
{

	double depthInMm, sqDist;
	Scalar sumOfWeights;
	vector<double> homDist, homWeights;
	vector<Point2f> tmpUndistRgbPoint, tmpUndistTPoint, tmpEstimatedPoint;
	Point2f estimatedPoint;
	Point tmpIdx;
	int MAP_TYPE = 0;
	int nbrCoordinates;
	int nbrClusters = int(stereoCalibParam.homDepthCentersRgb.size());
	int count = 0;
	int prevDepthInMm = stereoCalibParam.defaultDepth;

	// Are we mapping RGB or thermal points?
	if (vecUndistRgbCoord.size() > 0)
	{
		MAP_TYPE = 1;
		nbrCoordinates = int(vecUndistRgbCoord.size());
		vecUndistTCoord.clear();
	} else if (vecUndistTCoord.size() > 0)
	{
		MAP_TYPE = 2;
		nbrCoordinates = int(vecUndistTCoord.size());
		vecUndistRgbCoord.clear();
	} else
	{
		MAP_TYPE = 0;
		cerr << "Error in computeHomographyMapping - no coordinates to map!";
		return;
	}

	for (int i = 0; i < nbrCoordinates; i++)
	{
		if (vecDepthInMm.size() == 0) // Do we need to look up the depth?
		{
			if (vecDCoord.size() > 0)  // Are we providing any depth coordinates (as in the case of thermal coordinates)
			{
				if ((vecDCoord[i].y > 0) && (vecDCoord[i].x > 0) && (vecDCoord[i].y <= stereoCalibParam.HEIGHT) 
						&& (vecDCoord[i].x <= stereoCalibParam.WIDTH))
				{
					depthInMm =  lookUpDepth(dImg, vecDCoord[i], true);

					if (depthInMm > 7000)
					{ // Depth is undefined
						depthInMm = prevDepthInMm;
					}
				} else {
					depthInMm = prevDepthInMm;
				}
			} else {
				depthInMm = prevDepthInMm;
			}
		} else
		{
			// If not, the depth has previously been determined
			depthInMm = float(vecDepthInMm[i]);
		}

		prevDepthInMm = int(depthInMm);

		// Find the euclidean distances from the point to the homographies
		Mat worldCoordPoint(1,3,CV_32F,Scalar(0));
		Point3f tmpWorldCoord;
		

		// First, map the image coordinates to world coordinates
		if (MAP_TYPE == 1){
			worldCoordPoint.at<float>(0,0) = backProjectPoint(vecUndistRgbCoord[i].x, float(stereoCalibParam.rgbCamMat.at<double>(0,0)), 
							float(stereoCalibParam.rgbCamMat.at<double>(0,2)), float(depthInMm));
			worldCoordPoint.at<float>(0,1) = backProjectPoint(vecUndistRgbCoord[i].y, float(stereoCalibParam.rgbCamMat.at<double>(1,1)), 
							float(stereoCalibParam.rgbCamMat.at<double>(1,2)), float(depthInMm));
			worldCoordPoint.at<float>(0,2) = float(depthInMm);
		} else {
			worldCoordPoint.at<float>(0,0) = backProjectPoint(vecUndistTCoord[i].x, float(stereoCalibParam.tCamMat.at<double>(0,0)), 
							float(stereoCalibParam.tCamMat.at<double>(0,2)), 1500);
			worldCoordPoint.at<float>(0,1) = backProjectPoint(vecUndistTCoord[i].y, float(stereoCalibParam.tCamMat.at<double>(1,1)), 
							float(stereoCalibParam.tCamMat.at<double>(1,2)), 1500);
			worldCoordPoint.at<float>(0,2) = 1500;

		}

		tmpWorldCoord.x = worldCoordPoint.at<float>(0,0);
		tmpWorldCoord.y = worldCoordPoint.at<float>(0,1);
		tmpWorldCoord.z = worldCoordPoint.at<float>(0,2);
		worldCoordPointVector.push_back(tmpWorldCoord);

		int bestHomTmp = 0; // Reset parameters
		double bestDist = 1e6; 
		homDist.clear(); homWeights.clear(); 

		for (int j = 0; j < nbrClusters; j++)
		{
			if (stereoCalibParam.homDepthCentersRgb[j].z >= 0) // Is the current cluster valid? (also applies for thermal clusters)
			{ 
				Mat depthCenter(1,3,CV_32F,Scalar(0));
				// Compute Euclidean distance
				if (MAP_TYPE == 1)
				{
					homDist.push_back(sqrt(pow(worldCoordPoint.at<float>(0,0) - stereoCalibParam.homDepthCentersRgb[j].x, 2) +
						pow(worldCoordPoint.at<float>(0,1)- stereoCalibParam.homDepthCentersRgb[j].y, 2) +
						pow(worldCoordPoint.at<float>(0,2) - stereoCalibParam.homDepthCentersRgb[j].z, 2)));  // Euclidean					
				} else {
					homDist.push_back(sqrt(pow(worldCoordPoint.at<float>(0,1) - stereoCalibParam.homDepthCentersT[j].x, 2) +
						pow(worldCoordPoint.at<float>(0,1) - stereoCalibParam.homDepthCentersT[j].y, 2) +
						pow(worldCoordPoint.at<float>(0,2) - stereoCalibParam.homDepthCentersT[j].z, 2)));  // Euclidean
				}

				for (size_t k = 0; k < settings.discardedHomographies.size(); ++k) {
					if (j == settings.discardedHomographies[k]) {
						homDist[j] = 1e12;
					}
				}

				sqDist = homDist[j];
				if (sqDist < bestDist)
				{
					bestDist = sqDist;
					bestHomTmp = j;
				}
	
			} else {
				homDist.push_back(1e12);
			}
		}

		/* Compute the corresponding point in the other modality
			This includes the weighting of multiple homographies
		*/
		tmpUndistRgbPoint.clear(); tmpUndistTPoint.clear();
		estimatedPoint.x = 0; estimatedPoint.y = 0;
		if (MAP_TYPE == 1) {
			tmpUndistRgbPoint.push_back(vecUndistRgbCoord[i]);
		} else {
			tmpUndistTPoint.push_back(vecUndistTCoord[i]);
		}

		vector<int> octantIndicesTmp; vector<double> octantDistancesTmp;

		if (MAP_TYPE == 1) {
			trilinearInterpolator(tmpWorldCoord, stereoCalibParam.homDepthCentersRgb, homDist, homWeights, 
									octantIndicesTmp, octantDistancesTmp);

			weightedHomographyMapper(tmpUndistRgbPoint, tmpEstimatedPoint, stereoCalibParam.planarHom, homWeights);

		} else {
			trilinearInterpolator(tmpWorldCoord, stereoCalibParam.homDepthCentersT, homDist, homWeights, 
									octantIndicesTmp, octantDistancesTmp);

			weightedHomographyMapper(tmpUndistTPoint, tmpEstimatedPoint, stereoCalibParam.planarHomInv, homWeights);
		}

		octantIndices.push_back(octantIndicesTmp);
		octantDistances.push_back(octantDistancesTmp);
		estimatedPoint = tmpEstimatedPoint[0];

		if (MAP_TYPE == 1) {
			vecUndistTCoord.push_back(estimatedPoint);
		} else {
			vecUndistRgbCoord.push_back(estimatedPoint);
		}

		minDist.push_back(bestDist);
		bestHom.push_back(bestHomTmp);
	}

	if (count > 0) {
		std::cout << endl;
	}
}

void Registrator::trilinearInterpolator(Point3f inputPoint, vector<Point3f> &sourcePoints, vector<double> &precomputedDistance, vector<double> &weights, 
						   vector<int> &nearestSrcPointInd, vector<double> &nearestSrcPointDist)
{
	/*	TrilinearHomographyInterpolator finds the nearest point for each quadrant in 3D space and calculates weights
		based on trilinear interpolation for the input 3D point. The function returns a list of weights of the points
		used for the interpolation
	*/

	/* Octant map:
	I:		+ + +
	II:		- + +
	III:	- - +
	IV:		+ - +
	V:		+ + -
	VI:		- + -
	VII:	- - -
	VIII:	+ - -
	*/
	weights.clear();
	nearestSrcPointDist.clear();
	nearestSrcPointInd.clear();


	// Step 1: Label each sourcePoint according to the octant it belongs to
	vector<int> octantMap;

	for (size_t i = 0; i < sourcePoints.size(); ++i)
	{

		if ((sourcePoints[i].x - inputPoint.x) > 0)
		{	// x is positive; We are either in the first, fourth, fifth, or eighth octant
			
			if ((sourcePoints[i].y - inputPoint.y) > 0)
			{ // y is positive; We are either in the first or fifth octant

				if ((sourcePoints[i].z - inputPoint.z) > 0)
				{ // z is positive; We are in the first octant
					octantMap.push_back(1);
				} else {
					// z is negative; We are in the fifth octant				
					octantMap.push_back(5);
				}
			} else {
				// y is negative; We are either in the fourth or eighth octant
				
				if ((sourcePoints[i].z - inputPoint.z) > 0)
				{ // z is positive; We are in the fourth octant
					octantMap.push_back(4);
				} else {
					// z is negative; We are in the eighth octant				
					octantMap.push_back(8);
				}
			}
		} else {
			// x is negative; We are either in the second, third, sixth, or seventh octant
			
			if ((sourcePoints[i].y - inputPoint.y) > 0)
			{ // y is positive; We are either in the second or sixth octant

				if ((sourcePoints[i].z - inputPoint.z) > 0)
				{ // z is positive; We are in the second octant
					octantMap.push_back(2);
				} else {
					// z is negative; We are in the sixth octant				
					octantMap.push_back(6);
				}
			} else {
				// y is negative; We are either in the third or seventh octant
				
				if ((sourcePoints[i].z - inputPoint.z) > 0)
				{ // z is positive; We are in the third octant
					octantMap.push_back(3);
				} else {
					// z is negative; We are in the seventh octant				
					octantMap.push_back(7);
				}
			}
		}
	}

	// Step 2: Find the nearest point for every octant

	for (int i = 0; i < 8; ++i)
	{
		nearestSrcPointInd.push_back(-1);
		nearestSrcPointDist.push_back(1e6);
	}

	int currentOctant; 
	double currentDist;

	for (size_t i = 0; i < sourcePoints.size(); ++i)
	{
		// Identify the current octant
		currentOctant = octantMap[i];

		// Identify the distance to the input point
		currentDist = precomputedDistance[i];

		if (nearestSrcPointDist[currentOctant-1] > currentDist)
		{
			nearestSrcPointInd[currentOctant-1] = i;
			nearestSrcPointDist[currentOctant-1] = currentDist;
		}
	}

	// Step 3: Compute the relative weight of the eight surrounding points
	vector<double> unNormWeights;
	double tmpWeight;
	double sumOfDistances = 0;

	for (size_t i = 0; i < nearestSrcPointDist.size(); ++i)
	{
		if (nearestSrcPointDist[i] < 1e6)
		sumOfDistances += nearestSrcPointDist[i];
	}

	for (size_t i = 0; i < nearestSrcPointDist.size(); ++i)
	{
		if (nearestSrcPointDist[i] < 1e6)
		{
			tmpWeight = sumOfDistances / nearestSrcPointDist[i];
			unNormWeights.push_back(tmpWeight);
		} else {
			unNormWeights.push_back(0.);
		}
	}

	// Step 4: Normalize the weights
	double sumOfWeights = 0;
	vector<double> tmpWeightsVec;

	for (size_t i = 0; i < nearestSrcPointDist.size(); ++i)
	{
		sumOfWeights += unNormWeights[i];
	}

	for (size_t i = 0; i < nearestSrcPointDist.size(); ++i)
	{
		tmpWeight = unNormWeights[i] / sumOfWeights;
		tmpWeightsVec.push_back(tmpWeight);

		nearestSrcPointDist[i] = tmpWeight;
	}

	// Step 5: Recreate the weights to make a weight for every distance in precomputedDistance, even if the weight is zero
	for (size_t i = 0; i < precomputedDistance.size(); ++i)
	{
		weights.push_back(0.);
	}

	for (size_t i = 0; i < tmpWeightsVec.size(); ++i)
	{
		if (nearestSrcPointInd[i] >= 0)
		{
			weights[nearestSrcPointInd[i]] = tmpWeightsVec[i];
		}
	}

}

void Registrator::weightedHomographyMapper(vector<Point2f> undistPoint, vector<Point2f> &estimatedPoint, vector<Mat> &homographies, vector<double> &homWeights)
{
	/* weightedHomographyMapper maps the undistPoint (undistorted point) based by a weighted sum of the provided homographies, which are weighted by homWeights
	*/
	assert(homWeights.size() == homographies.size());
	vector<Point2f> tmpEstimatedPoint; Point2f tmpPoint;
	estimatedPoint.clear();
	estimatedPoint.push_back(tmpPoint);
	
	for (size_t i = 0; i < homWeights.size(); ++i)
	{
		if (homWeights[i] > 0)
		{
			perspectiveTransform(undistPoint, tmpEstimatedPoint, homographies[i]);

			estimatedPoint[0].x = estimatedPoint[0].x + tmpEstimatedPoint[0].x * float(homWeights[i]);
			estimatedPoint[0].y = estimatedPoint[0].y + tmpEstimatedPoint[0].y * float(homWeights[i]);
		}
	}

}

void Registrator::depthOutlierRemovalLookup(vector<Point2f> vecDCoord, vector<int> &vecDepthInMm)
{
	// depthOutlierRemovalLookup looks up depth for the provided coordinates of vecDCoord, discards garbage values
	// and makes constraints on the output depth in order to provide a smoothed depth lookup and removal of outlier values

	int depthInMm, depthNaNThreshold, validDepthMeasCount = 0, sumDepthInMm  = 0;
	bool SCALE_TO_THEORETICAL = true;
	int avgDepthInMm;
	vecDepthInMm.clear();
	vector<int> rawDepthInMm;

	if (SCALE_TO_THEORETICAL) {
		depthNaNThreshold = int(stereoCalibParam.depthCoeffA*6500+stereoCalibParam.depthCoeffB);
	} else {
		depthNaNThreshold = 6500;
	}

	if (vecDCoord.size() > 0)
	{

		// Step 1: Look up the depth for the entire depth coordinate set
		for (size_t i = 0; i < vecDCoord.size(); ++i)
		{
			if (vecDCoord.size() > 0)  // Are we providing any depth coordinates (as in the case of thermal coordinates)
			{
				if ((vecDCoord[i].y > 0) && (vecDCoord[i].x > 0) && (vecDCoord[i].y <= stereoCalibParam.HEIGHT) 
						&& (vecDCoord[i].x <= stereoCalibParam.WIDTH))
				{
					depthInMm =  int(lookUpDepth(dImg, vecDCoord[i], SCALE_TO_THEORETICAL));


					if (depthInMm > depthNaNThreshold)
					{ // Depth is undefined
						depthInMm = depthNaNThreshold;
					} else {
						// We have a valid depth measurement
						sumDepthInMm += depthInMm;
						++validDepthMeasCount;
					}
				} else {
					// Depth is undefined
					depthInMm = depthNaNThreshold;
				}
			} else {
				cerr << "No depth coordinates provided";
			}

			rawDepthInMm.push_back(depthInMm);
		}

		// Step 2: Filter outliers. Identify the outliers by perfoming K-means clustering and filter them 
		// according to a threshold. As we assume, that the human contours are in the foreground, this must
		// correspond to the minimum depth measurement cluster. Other depth measurements should be within a range
		// this cluster, otherwise they may be classified as outliers


		Mat labels, centers;
		Mat depthMat = Mat::zeros(Size(1,rawDepthInMm.size()), CV_32F);
		int nbrClusters;
	
		if (validDepthMeasCount < vecDCoord.size()) {
			// If there exists undefined depth in our dataset, we will need an extra cluster
			// to contain this
			nbrClusters = settings.nbrClustersForDepthOutlierRemoval;
		} else {
			// Otherwise, we will go for two clusters
			nbrClusters = settings.nbrClustersForDepthOutlierRemoval-1;
		}
		
		if (vecDCoord.size() > nbrClusters) {
			for (size_t i = 0; i < vecDCoord.size(); ++i) {
				depthMat.at<float>(i,0) = float(rawDepthInMm[i]);
			}

			kmeans(depthMat, nbrClusters, labels, TermCriteria( CV_TERMCRIT_EPS, 10,0.5), 100, 
					KMEANS_RANDOM_CENTERS, centers);

			int minAvg = depthNaNThreshold;

			for (int i = 0; i < nbrClusters; ++i) {
				// Find the lowest mean, or centre
				if (centers.at<float>(i,0) < minAvg) {
					minAvg = centers.at<float>(i,0);
				}
			}
			avgDepthInMm = minAvg;

			vector<int> clustersToBeDiscarded;

			for (int i = 0; i < nbrClusters; ++i) {
				// If depths are more than depthProximityThreshold mm away from the human contour, they should be regarded as noise
				if (centers.at<float>(i,0) > (minAvg+settings.depthProximityThreshold)) {
					clustersToBeDiscarded.push_back(i);
				}
			}

			for (size_t i = 0; i < vecDCoord.size(); ++i) {
				// Replace the noisy depth values by the average
				int currentCentre = labels.at<int>(i,0);
		
				for (size_t j = 0; j < clustersToBeDiscarded.size(); ++j) {
					if (currentCentre == clustersToBeDiscarded[j]) {
						rawDepthInMm[i] = minAvg;
					}
				}
			}

			vector<int> labelsVec, centersVec; // For debug
			for (size_t i = 0; i < vecDCoord.size(); ++i) {
				labelsVec.push_back(labels.at<int>(i,0));
			}

			for (int i = 0; i < nbrClusters; ++i) {
				centersVec.push_back(int(centers.at<float>(i,0)));
			}
		} else {
			// If we are operating on a very small dataset, calculate the mean manually
			if (validDepthMeasCount > 0) {
				avgDepthInMm = sumDepthInMm/validDepthMeasCount;
			} else {
				avgDepthInMm = stereoCalibParam.defaultDepth;
			}
		}

		// Step 3: Let the data through a non-causal moving-average filter
		int windowLength = 4;
		int filteredValue, validCounts;

		for (size_t i = 0; i < vecDCoord.size(); ++i)
		{
			if (rawDepthInMm[i] != depthNaNThreshold) { // Extract the current value
				filteredValue = rawDepthInMm[i];
				validCounts = 1;
			} else {
				filteredValue = 0;
				validCounts = 0;
			}

			// Go backwards
			int j = i-1;
			int currentStep = 0;

			while ((currentStep < (windowLength/2)) && (j >= 0))
			{
				if (rawDepthInMm[j] != depthNaNThreshold) {
					// If we have a valid value, add it to the filtered value
					filteredValue += rawDepthInMm[j];
					validCounts++;
					currentStep++;
				}
				--j;
			}

			// Go forwards
			j = i+1;
			currentStep = 0;
		

			while ((currentStep < (windowLength/2)) && (j < vecDCoord.size()))
			{
				if (rawDepthInMm[j] != depthNaNThreshold) {
					// If we have a valid value, add it to the filtered value
					filteredValue += rawDepthInMm[j];
					validCounts++;
					currentStep++;
				}
				++j;
			}

			// Compute the smoothed value and insert it into the depth vector
			if (validCounts > 0) {
				vecDepthInMm.push_back(filteredValue/validCounts);
			} else {
				vecDepthInMm.push_back(avgDepthInMm);
			}
		}
	}
}

void Registrator::MyDistortPoints(std::vector<cv::Point2f> src, std::vector<cv::Point2f> & dst, 
                     const cv::Mat & cameraMatrix, const cv::Mat & distorsionMatrix)
{
	// Normalize points before entering the distortion process
	Mat zeroDist;
	zeroDist = Mat::zeros(1,5,CV_32F);
	std::vector<cv::Point2f> normalizedUndistPoints;
	undistortPoints(src,normalizedUndistPoints,cameraMatrix,zeroDist);
	
	src = normalizedUndistPoints;
	

	// Code thanks to http://stackoverflow.com/questions/10935882/opencv-camera-calibration-re-distort-points-with-camera-intrinsics-extrinsics
  dst.clear();
  float fx = float(cameraMatrix.at<double>(0,0));
  float fy = float(cameraMatrix.at<double>(1,1));
  float ux = float(cameraMatrix.at<double>(0,2));
  float uy = float(cameraMatrix.at<double>(1,2));

  float k1 = float(distorsionMatrix.at<double>(0, 0));
  float k2 = float(distorsionMatrix.at<double>(0, 1));
  float p1 = float(distorsionMatrix.at<double>(0, 2));
  float p2 = float(distorsionMatrix.at<double>(0, 3));
  float k3 = float(distorsionMatrix.at<double>(0, 4));
  //BOOST_FOREACH(const cv::Point2d &p, src)
  for (unsigned int i = 0; i < src.size(); i++)
  {
    const cv::Point2f &p = src[i];
    float x = p.x;
    float y = p.y;
    float xCorrected, yCorrected;
    //Step 1 : correct distorsion
    {     
      float r2 = x*x + y*y;
      //radial distorsion
      xCorrected = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
      yCorrected = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

      //tangential distorsion
      //The "Learning OpenCV" book is wrong here !!!
      //False equations from the "Learning OpenCv" book
      //xCorrected = xCorrected + (2. * p1 * y + p2 * (r2 + 2. * x * x)); 
      //yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x);
      //Correct formulae found at : http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
      xCorrected = xCorrected + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
      yCorrected = yCorrected + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);
    }
    //Step 2 : ideal coordinates => actual coordinates
    {
      xCorrected = xCorrected * float(fx) + float(ux);
      yCorrected = yCorrected * float(fy) + float(uy);
    }
    dst.push_back(cv::Point2d(xCorrected, yCorrected));
  }

}



void Registrator::loadMinCalibrationVars(string calFile)
{
	FileStorage fsStereo(calFile, FileStorage::READ);

	if (fsStereo.isOpened())
	{ // If the file exists, we may read the stereo camera calibration parameters

		// Intrinsic camera parameters and distortion parameters
		fsStereo["rgbCamMat"] >> stereoCalibParam.rgbCamMat;
		fsStereo["rgbDistCoeff"] >> stereoCalibParam.rgbDistCoeff;
		fsStereo["tCamMat"] >> stereoCalibParam.tCamMat;
		fsStereo["tDistCoeff"] >> stereoCalibParam.tDistCoeff;

		// Depth to RGB registration maps
		fsStereo["rgbToDCalX"] >> stereoCalibParam.rgbToDCalX;
		fsStereo["rgbToDCalY"] >> stereoCalibParam.rgbToDCalY;
		fsStereo["dToRgbCalX"] >> stereoCalibParam.dToRgbCalX;
		fsStereo["dToRgbCalY"] >> stereoCalibParam.dToRgbCalY;

		// Homographies
		fsStereo["planarHom"] >> stereoCalibParam.planarHom;
		fsStereo["planarHomInv"] >> stereoCalibParam.planarHomInv;
		
		// 3D representations of the "centres" of the homographies
		fsStereo["homDepthCentersRgb"] >> stereoCalibParam.homDepthCentersRgb;
		fsStereo["homDepthCentersT"] >> stereoCalibParam.homDepthCentersT;
		
		// Misc parameters
		fsStereo["depthCoeffA"] >> stereoCalibParam.depthCoeffA;
		fsStereo["depthCoeffB"] >> stereoCalibParam.depthCoeffB;
		fsStereo["defaultDepth"] >> stereoCalibParam.defaultDepth;
		fsStereo["WIDTH"] >> stereoCalibParam.WIDTH;
		fsStereo["HEIGHT"] >> stereoCalibParam.HEIGHT;

		// Flags and settings
		fsStereo["UNDISTORT_IMAGES"] >> settings.UNDISTORT_IMAGES;
		fsStereo["depthProximityThreshold"] >> settings.depthProximityThreshold;
		fsStereo["nbrClustersForDepthOutlierRemoval"] >> settings.nbrClustersForDepthOutlierRemoval;

		if (settings.nbrClustersForDepthOutlierRemoval == 0)
		{
			settings.nbrClustersForDepthOutlierRemoval = 3;
		}

		fsStereo["discardedHomographies"] >> settings.discardedHomographies;
	}
	fsStereo.release();
}

void Registrator::drawRegisteredContours(cv::Mat rgbContourImage, cv::Mat& depthContourImage, cv::Mat& thermalContourImage, cv::Mat depthImg, bool preserveColors)
{
    // This functions draws the contours in rgbContourImage in a registered fashion in depthContourImage and thermalContourImage.
    // Note that in the current version, this does not handle contours inside holes in contours - although one-level holes in
    // contours are handled just fine. This is due to CV_RETR_CCOMP. For complete support, CV_RETR_TREE should be used.

    vector< vector<Point> > contourPoints;
    vector<Vec4i> hierarchy;
    Mat workImg = rgbContourImage.clone();	
	Mat erodedRgbContourImage = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContourImage.type());
	Mat gradX, gradY, grad;
	Mat absGradX, absGradY;

    // Find the contours to draw:
    findContours(workImg, contourPoints, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// Use the Sobel operator to find gradients of the image in order to shrink the contours
	Sobel(workImg, gradX, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(workImg, gradY, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	

    // Draw them, one at a time:
    for(int i = 0; i < contourPoints.size(); i++) {
        
		// Shrink the RGB contour to enhance the depth lookup. First, we need to compute the direction
		// of the intensity gradient from the Sobel derivatives
		vector<float> gradAngle;
		vector<Point> erodedContourPoints;
		int erosionSize = 5;
		int successCount = 0;
		Point prevPoint = contourPoints[i][0];

		for (size_t j = 0; j < contourPoints[i].size(); ++j) {
			float tmpAngle;
			tmpAngle = fastAtan2(gradX.at<float>(contourPoints[i][j]), gradY.at<float>(contourPoints[i][j]));
			gradAngle.push_back(tmpAngle);

			Point tmpErodedPoint;
			tmpErodedPoint.x = contourPoints[i][j].x-cos(tmpAngle)*erosionSize;
			tmpErodedPoint.y = contourPoints[i][j].y+sin(tmpAngle)*erosionSize;

			// Check if the eroded point is inside the mask
			if ((tmpErodedPoint.x < stereoCalibParam.WIDTH) && (tmpErodedPoint.y < stereoCalibParam.HEIGHT) &&
					(tmpErodedPoint.x >= 0) && (tmpErodedPoint.y >= 0) &&
					(rgbContourImage.at<uchar>(tmpErodedPoint) != 0)) {
				erodedContourPoints.push_back(tmpErodedPoint);
				prevPoint = tmpErodedPoint;
				successCount++;
			} else {
				// Otherwise, use the previous point
				erodedContourPoints.push_back(prevPoint);
			}
			vector< vector<Point> > drawPoints;
		}
		
		vector<Point> dContour, tContour;
        getRegisteredContours(contourPoints[i], erodedContourPoints, dContour, tContour, depthImg);

        vector< vector<Point> > drawPoints;

        drawPoints.push_back(dContour);
        Scalar color;
        if(preserveColors) {
            color = Scalar::all(rgbContourImage.at<uchar>(contourPoints[i][0]));
        } else {
            color = Scalar::all(255); // Draw white as default.
        }
        if(hierarchy[i][3] != -1) { // Has parent. Thus, is a hole. Draw black.
            color = Scalar::all(0);
        }
        drawContours(depthContourImage, drawPoints, 0, color, CV_FILLED);

		//drawPoints.clear();
		//drawPoints.push_back(erodedContourPoints);
		//drawContours(erodedRgbContourImage, drawPoints, 0, color, CV_FILLED);

        drawPoints.clear();
        drawPoints.push_back(tContour);
        drawContours(thermalContourImage, drawPoints, 0, color, CV_FILLED);
    }

	//imshow("erodedRgbContour", erodedRgbContourImage);
}

void Registrator::saveRegisteredContours(Mat depthContourImage, Mat thermalContourImage, string depthSavePath, string thermalSavePath, string imgNbr)
{
	// This function saves the contours from drawRegisteredContours

	// Append the proper number and file extension for the save path
	depthSavePath.append(imgNbr);
	thermalSavePath.append(imgNbr);

	imwrite(depthSavePath, depthContourImage);
	imwrite(thermalSavePath, thermalContourImage);
}

void Registrator::buildContourDirectory(string rgbLoadPath, vector<string> &rgbIndices)
{
	// Load contour images in RGB folder using dirent.h
	DIR *dr;
	int count = 0,len; 
	struct dirent *ent;
	const char * cRgbPath = rgbLoadPath.c_str();
	Mat tmpContourImg;

	dr = opendir(cRgbPath);
	if (dr != NULL) {
		while ((ent = readdir(dr)) != NULL) { 
			// While we have not reached the end of the folder, proceed
			len = int(strlen(ent ->d_name));
			if (len >= 4) {
				if (strcmp (".png",&(ent->d_name[len-4])) == 0) {
					rgbIndices.push_back(ent->d_name);
				}
			}
		}
	}

}

void Registrator::saveRegisteredImages(Mat depthImage, string depthSavePath, string imgNbr)
{
	// This function saves the contours from drawRegisteredContours

	// Append the proper number and file extension for the save path
	depthSavePath.append(imgNbr);
	imwrite(depthSavePath, depthImage);
}

void Registrator::buildImageDirectory(string rgbLoadPath, vector<string>& rgbIndices)
{
	string filetype = ".jpg";
	buildImageDirectory(rgbLoadPath, rgbIndices, filetype);
}

void Registrator::buildImageDirectory(string rgbLoadPath, vector<string>& rgbIndices, string filetype)
{
	DIR *dr;
	int count = 0,len; 
	struct dirent *ent;
	const char * cRgbPath = rgbLoadPath.c_str();
	const char * fType = filetype.c_str();
	rgbIndices.clear();

	dr = opendir(cRgbPath);
	if (dr != NULL) {
		while ((ent = readdir(dr)) != NULL) { 
			// While we have not reached the end of the folder, proceed
			len = int(strlen(ent ->d_name));
			if (len >= 4) {
				if (strcmp (fType,&(ent->d_name[len-4])) == 0) {
					rgbIndices.push_back(ent->d_name);
				}
			}
		}
	}
}

void Registrator::transformRgbImageToDepth(Mat rgbSourceImg, Mat &dDestImg, vector<Point2f> rgbPoints, vector<Point2f> dPoints)
{

	for (size_t i = 0; i < dPoints.size(); ++i)
	{
		if (rgbPoints[i].x >= 1 && rgbPoints[i].x < rgbSourceImg.cols && rgbPoints[i].y >= 1 && rgbPoints[i].y < rgbSourceImg.rows)
		{
			dDestImg.at<uchar>(int(dPoints[i].y-1), int(dPoints[i].x-1)) = rgbSourceImg.at<uchar>(int(rgbPoints[i].y-1), int(rgbPoints[i].x-1));
		}

	}

}

void Registrator::transformDepthImageToRgb(Mat dSourceImg, Mat &rgbDestImg, vector<Point2f> dPoints, vector<Point2f> rgbPoints)
{
	for (size_t i = 0; i < rgbPoints.size(); ++i)
	{
		if (dPoints[i].x >= 1 && dPoints[i].x < dSourceImg.cols && dPoints[i].y >= 1 && dPoints[i].y < dSourceImg.rows)
		{
			rgbDestImg.at<ushort>(rgbPoints[i].y-1, rgbPoints[i].x-1) = dSourceImg.at<ushort>(dPoints[i].y-1, dPoints[i].x-1);
		}

	}

}

void Registrator::loadTransformRgb()
{
	// This function loads RGB images, and one by one, registers them pixel-by-pixel to overlay them onto a depth image
	vector<string> rgbIndices;
	Mat rgbImage;

	string rgbImagePath = "P:/Private/Dataset/Annotation-Select/Scene 2/SyncRGB/";
	string depthImagePath = "P:/Private/Dataset/Annotation-Select/Scene 2/RegRgbToD/";

	buildImageDirectory(rgbImagePath, rgbIndices);
	vector<Point2f> dPoints, rgbPoints;

	for (size_t i = 0; i < rgbIndices.size(); ++i)
	{
		// Load the rgb image
		string tmpRgbPath = rgbImagePath;
		tmpRgbPath.append(rgbIndices[i]);
		rgbImage = imread(tmpRgbPath);

		if (i == 0)
		{
			for (size_t y = 1; y <= rgbImage.rows; ++y)
			{
				for (size_t x = 1; x <= rgbImage.cols; ++x)
				{
					Point2f dPoint(x,y);	
					dPoints.push_back(dPoint);
				}
			}

			computeCorrespondingRgbPointFromDepth(dPoints, rgbPoints);
		}

		// Create a "registered" rgb image as if was captured by the depth camera
		Mat depthImage = Mat::zeros(rgbImage.rows, rgbImage.cols, CV_8UC3);
		transformRgbImageToDepth(rgbImage, depthImage, rgbPoints, dPoints);

		saveRegisteredImages(depthImage, depthImagePath, rgbIndices[i]);
		cout << i << " ";

	}

}

void Registrator::loadTransformDepth()
{
	// This function loads RGB images, and one by one, registers them pixel-by-pixel to overlay them onto a depth image
	vector<string> depthIndices;
	Mat depthImage;

	string depthImagePath = "P:/Private/Dataset/Annotation-Select/Scene 2/SyncD/";
	string rgbImagePath = "P:/Private/Dataset/Annotation-Select/Scene 2/RegDToRgb/";

	buildImageDirectory(depthImagePath, depthIndices,".png");
	vector<Point2f> dPoints, rgbPoints;

	for (size_t i = 0; i < depthIndices.size(); ++i)
	{
		// Load the depth image
		string tmpDepthPath = depthImagePath;
		tmpDepthPath.append(depthIndices[i]);
		depthImage = imread(tmpDepthPath,CV_LOAD_IMAGE_ANYDEPTH);

		if (i == 0)
		{
			for (int y = 1; y <= depthImage.rows; ++y)
			{
				for (int x = 1; x <= depthImage.cols; ++x)
				{
					Point2f rgbPoint(x,y);	
					rgbPoints.push_back(rgbPoint);
				}
			}

			computeCorrespondingDepthPointFromRgb(rgbPoints, dPoints);
		}

		// Create a "registered" rgb image as if was captured by the depth camera
		Mat rgbImage = Mat::zeros(depthImage.rows, depthImage.cols, depthImage.type());
		transformDepthImageToRgb(depthImage, rgbImage, dPoints, rgbPoints);

		saveRegisteredImages(rgbImage, rgbImagePath, depthIndices[i]);
		std::cout << i << " ";

	}

}

void Registrator::loadRegSaveContours()
{
    const char* rgbContourPath = "Data/Masks/Scene1/RGBMasks/";
	const char* depthContourPath = "Data/Masks/Scene1/DepthMasks/";
	const char* thermalContourPath = "Data/Masks/Scene1/ThermalMasks/";
	const char* depthImagePath = "Data/Sequences/Scene1/SyncD/";
    
	this->loadRegSaveContours(rgbContourPath, depthContourPath, thermalContourPath, depthImagePath);
}

void Registrator::loadRegSaveContours(vector<Mat> masksColor, vector<Mat> masksDepth,
                                      vector<Mat> & masksThermal, vector<Mat> depth) {
    //This function uses a list of RGB contours already read and, one by one, registers them
    //in the thermal and depth modalities
    for (size_t i = 0; i < masksColor.size(); i++ ) {
        //Rgb contour and corresponding depth image
        Mat rgbContour = masksColor[i];
        Mat depthImage = depth[i];
        
        //Create empty thermal and depth images
        Mat thermalContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());
        Mat	depthContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());
        
        // Register the thermal and depth contours
        drawRegisteredContours(rgbContour, depthContour, thermalContour, depthImage, true);
        imshow("rgbContour", rgbContour);
        //imshow("depthContour",depthContour);
        imshow("thermalContour",thermalContour);
        imshow("depthImage", depthImage);
        if(waitKey(30)>= 0)  {}
        //cout << rgbIndices[i] << ",";
        
        masksThermal.push_back(thermalContour);
    }
}


void Registrator::loadRegSaveContours(const char* pathMasksColor, const char* pathMasksDepth,
                                      const char* pathMasksThermal, const char* pathDepth)
{
	// This function loads a list of RGB contours, and, one by one, reads RGB contours, and
	// registers them in the thermal and depth modalities
    vector<string> rgbIndices;
	Mat rgbContour, depthImage;
    
	string rgbContourPath = pathMasksColor;
	string depthContourPath = pathMasksDepth;
	string thermalContourPath = pathMasksThermal;
	string depthImagePath = pathDepth;
    
	this->buildContourDirectory(rgbContourPath, rgbIndices);

    for (size_t i = 0; i < rgbIndices.size(); ++i) {
        
        // Load the rgb contour and corresponding depth image
		string tmpRgbPath = rgbContourPath;
		tmpRgbPath.append(rgbIndices[i]);
		rgbContour = imread(tmpRgbPath, CV_LOAD_IMAGE_GRAYSCALE);
        
		string tmpDepthPath = depthImagePath;
		tmpDepthPath.append(rgbIndices[i]);
		depthImage = imread(tmpDepthPath, CV_LOAD_IMAGE_ANYDEPTH);
        
		// Create empty thermal and depth images
		Mat thermalContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());
		Mat	depthContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());
        
		// Register the thermal and depth contours
		drawRegisteredContours(rgbContour, depthContour, thermalContour, depthImage, true);
		//imshow("rgbContour", rgbContour);
		//imshow("depthContour",depthContour);
		//imshow("thermalContour",thermalContour);
        
		//cvWaitKey(0);
		cout << rgbIndices[i] << ",";
        
		// Save the contours
		saveRegisteredContours(depthContour, thermalContour, depthContourPath, thermalContourPath, rgbIndices[i]);
    }
    
}

/*void Registrator::loadRegSaveContours()
{
	// This function loads a list of RGB contours, and, one by one, reads RGB contours, and 
	// registers them in the thermal and depth modalities

	vector<string> rgbIndices;
	Mat rgbContour, depthImage;

	string rgbContourPath = "C:/Users/Chris Bahnsen/Desktop/trimodal/color-depth/";
	string depthContourPath = "C:/Users/Chris Bahnsen/Desktop/trimodal/backprojected depth/";
	string thermalContourPath = "C:/Users/Chris Bahnsen/Desktop/trimodal/newThermal/";
	string depthImagePath = "C:/Users/Chris Bahnsen/Desktop/trimodal/raw depth frames/";

	this->buildContourDirectory(rgbContourPath, rgbIndices);

	for (size_t i = 0; i < rgbIndices.size(); ++i) {
		
		// Load the rgb contour and corresponding depth image
		string tmpRgbPath = rgbContourPath;
		tmpRgbPath.append(rgbIndices[i]);
		rgbContour = imread(tmpRgbPath, CV_LOAD_IMAGE_GRAYSCALE);

		string tmpDepthPath = depthImagePath;
		tmpDepthPath.append(rgbIndices[i]);
		depthImage = imread(tmpDepthPath, CV_LOAD_IMAGE_ANYDEPTH);

		// Create empty thermal and depth images
		Mat thermalContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());
		Mat	depthContour = Mat::zeros(Size(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT), rgbContour.type());

		// Register the thermal and depth contours
		drawRegisteredContours(rgbContour, depthContour, thermalContour, depthImage, true);
		//imshow("rgbContour", rgbContour);
		//imshow("depthContour",depthContour);
		//imshow("thermalContour",thermalContour);

		//cvWaitKey(0);
		cout << rgbIndices[i] << ",";

		// Save the contours
		saveRegisteredContours(depthContour, thermalContour, depthContourPath, thermalContourPath, rgbIndices[i]);
	}
}*/




void Registrator::getRegisteredContours(vector<Point> contour, vector<Point> erodedContour, vector<Point>& dContour, vector<Point>& tContour, Mat depthImg)
{
    vector<Point2f> rgbCoords, erodedRgbCoords, dCoords, erodedDCoords, tCoords;
	depthImg.copyTo(this->dImg); // Required for the registration of RGB to thermal

    for(int j = 0; j < contour.size(); j++) {
        rgbCoords.push_back(Point(contour[j].x, contour[j].y));
    }

	for(int j = 0; j < erodedContour.size(); j++) {
        erodedRgbCoords.push_back(Point(erodedContour[j].x, erodedContour[j].y));
    }

    this->computeCorrespondingDepthPointFromRgb(erodedRgbCoords, erodedDCoords);
	this->computeCorrespondingDepthPointFromRgb(rgbCoords, dCoords);
    for(int j = dCoords.size()-1; j >= 0; j--) {
		
		if ((dCoords[j].x == 0) && (dCoords[j].y == 0)) {
			// Depth coordinate is not valid. Remove corresponding rgbCoord and erodedDCoord
			rgbCoords.erase(rgbCoords.begin()+j);
			erodedDCoords.erase(erodedDCoords.begin()+j);
		} else{
			// Depth coordinate is valid. Insert from beginning
			dContour.insert(dContour.begin(),(Point(dCoords[j].x, dCoords[j].y)));
		}
    }

    vector<int> homInd, vecDepthInMm;
	vector<double> minDist;
	vector<vector<double>> octantDistances;
	vector<vector<int>> octantIndices;
	vector<Point3f> worldCoordPointVector;

	this->depthOutlierRemovalLookup(erodedDCoords, vecDepthInMm); // Use the coordinates of the eroded mask to look up the depth


	if (rgbCoords.size() > 0) {
		this->computeCorrespondingThermalPointFromRgb(rgbCoords, tCoords, dCoords, vecDepthInMm, minDist, 
												homInd, octantIndices, octantDistances, worldCoordPointVector);
	}

    for(int j = 0; j < tCoords.size(); j++) {
        tContour.push_back(Point(tCoords[j].x, tCoords[j].y));
    }
}


// **** The remaining code is functions to show images and handle mouse callbacks in OpenCV
void Registrator::markCorrespondingPointsRgb(Point2f rgbCoord)
{
	vector<Point2f> vecTCoord,vecDCoord,epiTCoord,vecDistRgbCoord,vecRgbCoord,tempVecRgbCoord;
	Point2f tCoord,dCoord,distRgbCoord;
	Mat tmpDImg;
	vecRgbCoord.push_back(rgbCoord);
	vector<int> homInd;

	// ***** Compute the corresponding point in the depth image ******
	computeCorrespondingDepthPointFromRgb(vecRgbCoord,vecDCoord);
	dCoord = vecDCoord[0];

	// ***** Compute the corresponding point in the thermal image *****
	computeCorrespondingThermalPointFromRgb(vecRgbCoord, vecTCoord, vecDCoord, homInd);

	tCoord = vecTCoord[0];

	if (homInd.size() == 0)
	{
		homInd.push_back(0);
	}

	// Mark the points in the RGB, thermal, and depth image
	markCorrespondingPoints(rgbCoord, tCoord, dCoord, homInd[0], MAP_FROM_RGB );

}

void Registrator::markCorrespondingPointsThermal(Point2f tCoord)
{
	vector<Point2f> vecTCoord,vecRgbCoord,epiCoord, vecDCoord;
	Point2f rgbCoord, dCoord;
	vecTCoord.push_back(tCoord);
	Mat homEpiToRgb;
	

	// Find the corresponding point in RGB coordinates and process the information to markCorrespondingPoints()
	vector<double> minDist; vector<int> bestHom;  vector<vector<int>> octantIndices; vector<vector<double>> octantDistances;
	computeCorrespondingRgbPointFromThermal(vecTCoord, vecRgbCoord, minDist, bestHom, octantIndices, octantDistances);
	rgbCoord = vecRgbCoord[0];
	
	computeCorrespondingDepthPointFromRgb(vecRgbCoord,vecDCoord);
	dCoord = vecDCoord[0];

	if (bestHom.size() == 0)
	{
		bestHom.push_back(0);
	}

	markCorrespondingPoints(rgbCoord, tCoord, dCoord, bestHom[0], MAP_FROM_THERMAL);
	
}

void Registrator::markCorrespondingPointsDepth(Point2f dCoord)
{
	// Find the corresponding point in RGB coordinates and process the information to markCorrespondingPointsRgb()
	Point2f rgbCoord, tCoord;
	vector<Point2f> vecDCoord,vecRgbCoord, vecTCoord;
	vecDCoord.push_back(dCoord);
	
	computeCorrespondingRgbPointFromDepth(vecDCoord, vecRgbCoord);
	rgbCoord = vecRgbCoord[0];
	
	// ***** Compute the corresponding point in the thermal image *****
	vector<int> bestHom;

	computeCorrespondingThermalPointFromRgb(vecRgbCoord, vecTCoord, vecDCoord, bestHom);
	tCoord = vecTCoord[0];

	markCorrespondingPoints(rgbCoord, tCoord, dCoord, bestHom[0], MAP_FROM_DEPTH);
}

void Registrator::markCorrespondingPoints(Point2f rgbCoord, Point2f tCoord, Point2f dCoord, int homInd, int MAP_TYPE)
{
	Scalar rgbColor(0,0,0);
	Scalar tColor(0,0,0);
	Scalar dColor(65535,65535,65535);

	rgbColor[3] = 255;

	// Define the color of the circles based on the origin of the marking
	switch (MAP_TYPE)
	{
		case 0: // Map from RGB
			rgbColor[1] = 255;
			tColor[2] = 255;
			//dColor[2] = 255;
			break;
		case 1: // Map from thermal
			rgbColor[2] = 255;
			tColor[1] = 255;
			//dColor[2] = 255;
			break;
		case 2: // Map from depth
			rgbColor[2] = 255;
			tColor[2] = 255;
			//dColor[1] = 255;
			break;
	}

	// Mark the points in the RGB, thermal, and depth image
	circle(rgbImg, rgbCoord, 2, rgbColor, -1);
	imshow("RGB",rgbImg);

	circle(tImg, tCoord, 2, tColor, -1);
	imshow("Thermal",tImg);

	//Mat tmpDImg;
	//dImg.copyTo(tmpDImg);
	circle(dImg, dCoord, 2, dColor, -1);
	imshow("Depth",dImg);

	float depthInMm;
	// Find the depth value of the point and write it in the status window
	if ((dCoord.y > 0) && (dCoord.x > 0) && (dCoord.y <= stereoCalibParam.HEIGHT) 
			&& (dCoord.x <= stereoCalibParam.WIDTH))
	{
		depthInMm =  lookUpDepth(dImg, dCoord, true);
	} else {
		depthInMm = 0;
	}


	// Write the coordinates in the status field
	char statusMsg[150];
	sprintf(statusMsg,"RGB: %3.0f,%3.0f :: Depth: %3.0f,%3.0f :: Thermal: %3.0f,%3.0f :: Depth of point: %4.1f mm :: Using hom #%d",rgbCoord.x,rgbCoord.y,dCoord.x,dCoord.y,tCoord.x,tCoord.y,depthInMm,homInd);

	displayStatusBar("RGB",statusMsg,0);
}

void Registrator::updateFrames(int imgNbr, void* obj)
{
	Registrator* minReg = (Registrator*) obj; // Recast
	
	if (imgNbr == 0) {
		imgNbr = 1; 
		setTrackbarPos("Frame","RGB",imgNbr);
	}
	minReg->showModalityImages(minReg->rgbImgPath,minReg->tImgPath,minReg->dImgPath,imgNbr);
}

void Registrator::showModalityImages(string rgbPath,string tPath,string dPath,int imgNbr)
{
	string rgbImgStr,tImgStr,dImgStr;
	Mat distTImg,distRgbImg,rectTImg,rectRgbImg;
	Size imgSize(stereoCalibParam.WIDTH,stereoCalibParam.HEIGHT);
	
	// Insert the number of leading zeros
	stringstream nbrStr;
	nbrStr << setfill('0') << setw(5) << imgNbr;

	rgbImgStr = rgbPath + nbrStr.str() + ".jpg";
	tImgStr = tPath + nbrStr.str() + ".jpg";
	dImgStr = dPath + nbrStr.str() + ".png";

	distTImg = imread(tImgStr); distRgbImg = imread(rgbImgStr);  // Read thermal and rgb images
	
	try
	{
		if (settings.UNDISTORT_IMAGES)
		{
			undistort(distTImg,rectTImg,stereoCalibParam.tCamMat,stereoCalibParam.tDistCoeff);
			undistort(distRgbImg,rectRgbImg,stereoCalibParam.rgbCamMat,stereoCalibParam.rgbDistCoeff);
			tImg = rectTImg;
			rgbImg = rectRgbImg;
		} else{
			tImg = distTImg;
			rgbImg = distRgbImg;
		}
	} 
	catch (exception& e){
		    cerr << "Unable to load images " << e.what() << endl;
	}

	try {
	imshow("Thermal",tImg); imshow("RGB",rgbImg); // Show thermal and rgb images
	dImg = imread(dImgStr,CV_LOAD_IMAGE_ANYDEPTH); imshow("Depth",dImg); // Read and show the depth image
	} catch (exception& e){
		cerr << "Unable to show images " << e.what() << endl;
	}
	
}

void Registrator::rgbOnMouse( int event, int x, int y, int)
{
	if( event != CV_EVENT_LBUTTONDOWN )
        return;

	Point2i rgbCoord;
	rgbCoord.x = x;
	rgbCoord.y = y;
		
	markCorrespondingPointsRgb(rgbCoord);
}

void Registrator::thermalOnMouse( int event, int x, int y, int)
{
	if( event != CV_EVENT_LBUTTONDOWN )
        return;

	Point2i tCoord;
	tCoord.x = x;
	tCoord.y = y;

	markCorrespondingPointsThermal(tCoord);

}

void Registrator::depthOnMouse( int event, int x, int y, int)
{
	if( event != CV_EVENT_LBUTTONDOWN )
        return;

	Point2i dCoord;
	dCoord.x = x;
	dCoord.y = y;
	markCorrespondingPointsDepth(dCoord);

}

void Registrator::initWindows(int imgNbr)
{
	// Create windows
	namedWindow("RGB",CV_WINDOW_AUTOSIZE);
	namedWindow("Thermal",CV_WINDOW_AUTOSIZE);
	namedWindow("Depth",CV_WINDOW_AUTOSIZE);

	// Get number of images in rgb folder
	DIR *dr;
	int count = 0,len; 
	struct dirent *pDirent;
	const char * cRgbPath = this->rgbImgPath.c_str();

	dr = opendir(cRgbPath);
	if (dr != NULL) {
		while ((pDirent = readdir(dr)) != NULL) {
			len = int(strlen(pDirent ->d_name));
			if (len >= 4) {
				if (strcmp (".jpg",&(pDirent->d_name[len-4])) == 0) {
					count++;
				}
			}
		}
	}
	
	// Create trackbar
	cvCreateTrackbar2("Frame","RGB",&imgNbr,count, Registrator::updateFrames, this);
	showModalityImages(rgbImgPath, tImgPath, dImgPath,imgNbr);
}

void wrappedRgbOnMouse(int event, int x, int y, int flags, void* ptr)
{
    Registrator* mcPtr = (Registrator*)ptr;
    if(mcPtr != NULL)
        mcPtr->rgbOnMouse(event, x, y, flags);
}

void wrappedThermalOnMouse(int event, int x, int y, int flags, void* ptr)
{
    Registrator* mcPtr = (Registrator*)ptr;
    if(mcPtr != NULL)
        mcPtr->thermalOnMouse(event, x, y, flags);
}

void wrappedDepthOnMouse(int event, int x, int y, int flags, void* ptr)
{
    Registrator* mcPtr = (Registrator*)ptr;
    if(mcPtr != NULL)
        mcPtr->depthOnMouse(event, x, y, flags);
}

string Registrator::getRgbImgPath()
{
	return this->rgbImgPath;
}

string Registrator::getTImgPath()
{
	return this->tImgPath;
}

string Registrator::getDImgPath()
{
	return this->dImgPath;
}

void Registrator::setRgbImgPath(string path)
{
	this->rgbImgPath = path;
}

void Registrator::setTImgPath(string path)
{
	this->tImgPath = path;
}

void Registrator::setDImgPath(string path)
{
	this->dImgPath = path;
}

void Registrator::toggleUndistortion(bool undistort)
{
	this->settings.UNDISTORT_IMAGES = undistort;
}

void Registrator::setDiscardedHomographies(vector<int> discardedHomographies) 
{
	this->settings.discardedHomographies = discardedHomographies;
}

void Registrator::setUsePrevDepthPoint(bool value)
{
	this->settings.USE_PREV_DEPTH_POINT = value;
}

/*int main(int argc)
{	
	// Create an instance of the registration class
	Registrator minReg;

	// Load the calibration and registration variables from the .yml-file
	minReg.loadMinCalibrationVars("P:/Private/Trimodal registration/Dataset/Registration Variables/Scene 1/calibVars.yml");

	// Set the paths of the trimodal imagery
	minReg.setRgbImgPath("C:/Users/Chris Bahnsen/Desktop/trimodal/color-depth/");
	minReg.setTImgPath("C:/Users/Chris Bahnsen/Desktop/trimodal/color-depth/");
	minReg.setDImgPath("C:/Users/Chris Bahnsen/Desktop/trimodal/color-depth/");
	
	// Are the input coordinates undistorted, or are we working on the original, distorted images? 
	// If input coordinates are distorted, this property should be set to false
	minReg.toggleUndistortion(false);

	// When registering contours, this property should be set to true - otherwise, it should be false
	minReg.setUsePrevDepthPoint(true);

	// Show the synchronized imagery in three different windows of OpenCV:
	int imgNbr = 1;
	//minReg.initWindows(imgNbr); 

	minReg.loadRegSaveContours();

	// Wrapper for handling tbe mouse events to the registrator class
	//setMouseCallback("RGB", wrappedRgbOnMouse, (void*)&minReg);
	//setMouseCallback("Thermal", wrappedThermalOnMouse, (void*)&minReg);
	//setMouseCallback("Depth", wrappedDepthOnMouse, (void*)&minReg);
	
	// Run forever
	while (true)
	{
		cvWaitKey(-1);
	}

	return 0;
}*/