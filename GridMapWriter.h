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

class GridMapWriter
{
public:
    GridMapWriter();
    GridMapWriter(ModalityGridData& mgd, GridMat& values);
    
    void setModalityGridData(ModalityGridData& mgd);
    void setGridCellValues(GridMat& values);
    
    template<typename T>
    void write(std::string dir);
    template<typename T>
    void write(ModalityGridData& mgd, GridMat& values, std::string dir);
    
private:
    ModalityGridData m_mgd;
    GridMat m_values;
      
    void loadFilenames(string dir, const char* filetype, vector<string>& filenames);

};

#endif /* defined(__segmenthreetion__GridMapWriter__) */
