//
//  GridMapWriter.h
//  segmenthreetion
//
//  Created by Albert Clapés on 07/03/14.
//
//

#ifndef __segmenthreetion__GridMapWriter__
#define __segmenthreetion__GridMapWriter__

#include <iostream>

#include "GridMat.h"
#include "ModalityGridData.hpp"

template<typename T>
class GridMapWriter
{
public:
    GridMapWriter();
    GridMapWriter(ModalityGridData mgd, int n, GridMat values);
    
    void setModalityGridData(ModalityGridData mgd);
    void setGridCellValues(GridMat values);
    void setNumberOfMaps(int n);
    
    void write(std::string path);
    void write(ModalityGridData& mgd, int n, GridMat& values, std::string path);
    
private:
    ModalityGridData m_mgd;
    GridMat m_values;
    int m_n;
};

#endif /* defined(__segmenthreetion__GridMapWriter__) */
