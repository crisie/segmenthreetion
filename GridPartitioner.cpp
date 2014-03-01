//
//  GridPartitioner.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 01/03/14.
//
//

#include "GridPartitioner.h"


GridPartitioner::GridPartitioner()
        : m_hp(2), m_wp(2)
{

}


GridPartitioner::GridPartitioner(unsigned int hp, unsigned int wp)
: m_hp(hp), m_wp(wp)
{
    
}


GridPartitioner::GridPartitioner(unsigned int hp, unsigned int wp, ModalityData md)
: m_hp(hp), m_wp(wp), m_md(md)
{
    
}


void GridPartitioner::setModalityData(ModalityData md)
{
    m_md = md;
}


void GridPartitioner::setGridPartitions(unsigned int hp, unsigned int wp)
{
    m_hp = hp;
    m_wp = wp;
}


void GridPartitioner::grid(ModalityGridData& mgd)
{
    grid(m_md, mgd);
}


void GridPartitioner::grid(ModalityData md, ModalityGridData& mgd)
{
    m_md = md;
    if (!m_md.isFilled())
        return; // do nothing
    
    vector<GridMat> gframes;
    vector<GridMat> gmasks;
    cv::Mat tags;
    
    // Grid frames and masks
    grid(md.getFrames(), md.getBoundingRects(), m_hp, m_wp, gframes);
    grid(md.getMasks(), md.getBoundingRects(), md.getTags(), m_hp, m_wp, gmasks, tags);
    //visualizeGridmats(gframes_train); // DEBUG
    //visualizeGridmats(gmasks_train); // DEBUG
}


/*
 * Trim subimages, defined by rects (bounding boxes), from image frames
 */
void GridPartitioner::grid(vector<cv::Mat>& images, vector< vector<cv::Rect> > rects, unsigned int crows, unsigned int ccols, vector<GridMat>& grids)
{
    //namedWindow("grided subject");
    // Seek in each frame ..
    for (unsigned int f = 0; f < rects.size(); f++)
    {
        // .. all the people appearing
        for (unsigned int r = 0; r < rects[f].size(); r++)
        {
            if (rects[f][r].height >= m_hp && rects[f][r].width >= m_wp)
            {
                cv::Mat subject (images[f], rects[f][r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject = (subject == (m_md.getMasksOffset() + r));
                subject.release();
                
                GridMat g (maskedSubject, crows, ccols);
                grids.push_back( g );
            }
        }
    }
}


/*
 * Trim subimages, defined by rects (bounding boxes), from image frames
 */
void GridPartitioner::grid(vector<cv::Mat>& images, vector< vector<cv::Rect> > rects, vector< vector<int> > rtags, unsigned int crows, unsigned int ccols, vector<GridMat>& grids, cv::Mat& tags)
{
    vector<int> tagsAux;
    
    // Seek in each frame ..
    for (unsigned int f = 0; f < rects.size(); f++)
    {
        // .. all the people appearing
        for (unsigned int r = 0; r < rects[f].size(); r++)
        {
            if (rects[f][r].height >= m_hp && rects[f][r].width >= m_wp)
            {
                cv::Mat subject (images[f], rects[f][r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject = (subject == (m_md.getMasksOffset() + r));
                subject.release();
                
                GridMat g (maskedSubject, crows, ccols);
                grids.push_back( g );
                
                tagsAux.push_back(rtags[f][r]);
            }
        }
    }
    
    cv::Mat tmp (tagsAux.size(), 1, cv::DataType<int>::type, tagsAux.data());
    tmp.copyTo(tags);
}