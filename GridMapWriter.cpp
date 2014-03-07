//
//  GridMapWriter.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 07/03/14.
//
//

#include "GridMapWriter.h"

template<typename T>
GridMapWriter<T>::GridMapWriter()
: m_n(0)
{
    
}

template<typename T>
GridMapWriter<T>::GridMapWriter(ModalityGridData mgd, int n, GridMat values)
: m_mgd(mgd), m_n(n), m_values(values)
{
    
}

template<typename T>
void GridMapWriter<T>::setModalityGridData(ModalityGridData mgd)
{
    m_mgd = mgd;
}

template<typename T>
void GridMapWriter<T>::setGridCellValues(GridMat values)
{
    m_values = values;
}

template<typename T>
void GridMapWriter<T>::setNumberOfMaps(int n)
{
    m_n = n;
}

template<typename T>
void GridMapWriter<T>::write(std::string path)
{
    if (n < 1)
    {
        std::cerr << "Data was not set. Set it or use the proper write function" << std::endl;
        return;
    }
    
    write(m_mgd, m_n, m_values, path);
}

template<typename T>
void GridMapWriter<T>::write(ModalityGridData& mgd, int n, GridMat& values, std::string path)
{
    for (int f = 1; f <= n; f++)
    {
        cv::Mat map (mgd.getFrameResolution(f).y, mgd.getFrameResolution(f).x, cv::DataType<T>::type);
        
        cv::Mat frameIndices;
        cv::findNonZero(mgd.getGridsFrameIDs() == f, frameIndices);
        for (int k = 0; k < frameIndices.rows; k++) // "g++", huhu :'D
        {
            for (int i = 0; i < mgd.hp(); i++) for (int j = 0; j < mgd.wp(); j++)
            {
                cv::Mat roiMask (map, mgd.getGridBoundingRect(k));
                map.setTo(values.at<T>(i, j, k, 0), roiMask);
            }
        }
        
        std::stringstream ss;
        ss << f << ".png";
        cv::imwrite(std::string(ss.str()), map);
    }
}