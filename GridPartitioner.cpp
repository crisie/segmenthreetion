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


void GridPartitioner::setGridPartitions(unsigned int hp, unsigned int wp)
{
    m_hp = hp;
    m_wp = wp;
}


void GridPartitioner::grid(ModalityData& md, ModalityGridData& mgd)
{
    if (!md.isFilled())
        return; // do nothing
    
    vector<GridMat> gframes;
    vector<GridMat> gmasks;
    cv::Mat gframeids;
    vector<cv::Rect> gboundingrects;
    cv::Mat tags;
    
//    // Grid frames and masks
    gridFrames(md, gframes, gframeids);
    gridMasks(md, gmasks, gboundingrects, tags);
    //visualizeGridmats(gframes_train); // DEBUG
    //visualizeGridmats(gmasks_train); // DEBUG
    
    mgd.setGridsFrames(gframes);
    mgd.setGridsMasks(gmasks);
    mgd.setGridsFrameIDs(gframeids);
    //mgd.setFramesResolutions(gresolutions);
    mgd.setGridsBoundingRects(gboundingrects);
    mgd.setTags(tags);
}

/*
 * Trim subimages, defined by rects (bounding boxes), from image frames
 */
void GridPartitioner::gridFrames(ModalityData& md, vector<GridMat>& gframes, cv::Mat& frameIDs)
{
    //namedWindow("grided subject");
    // Seek in each frame ..
    int ngrids = 0;
    for (unsigned int f = 0; f < md.getFrames().size(); f++)
    {
        vector<cv::Rect> rects = md.getPredictedBoundingRectsInFrame(f);
        // .. all the people appearing
        for (unsigned int r = 0; r < rects.size(); r++)
        {
            if (rects[r].height >= m_hp && rects[r].width >= m_wp)
            {
                cv::Mat subject (md.getFrame(f), rects[r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject;
                subject.copyTo(maskedSubject, md.getPredictedMask(f,r));
                subject.release();
                
                GridMat g (maskedSubject, m_hp, m_wp);
                gframes.push_back( g );
                
                frameIDs.at<int>(ngrids++, 0) = f;
            }
            
        }
    }
}


/*
 * Trim subimages, defined by rects (bounding boxes), from image frames
 */
void GridPartitioner::gridMasks(ModalityData& md, vector<GridMat>& gmasks)
{
    // Seek in each frame ..
    for (unsigned int f = 0; f < md.getFrames().size(); f++)
    {
        vector<cv::Rect> rects = md.getPredictedBoundingRectsInFrame(f);
        // .. all the people appearing
        for (unsigned int r = 0; r < rects.size(); r++)
        {
            if (rects[r].height >= m_hp && rects[r].width >= m_wp)
            {
                cv::Mat subject (md.getFrame(f), rects[r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject;
                subject.copyTo(maskedSubject, md.getPredictedMask(f,r));
                subject.release();
                
                GridMat g (maskedSubject, m_hp, m_wp);
                gmasks.push_back( g );
            }
        }
    }
}

void GridPartitioner::gridMasks(ModalityData& md, vector<GridMat>& gmasks, vector<cv::Rect>& grects, cv::Mat& gtags)
{
    vector<int> tagsAux;
    
    // Seek in each frame ..
    for (unsigned int f = 0; f < md.getFrames().size(); f++)
    {
        vector<cv::Rect> rects = md.getPredictedBoundingRectsInFrame(f);
        vector<int> tags = md.getTagsInFrame(f);
        // .. all the people appearing
        for (unsigned int r = 0; r < rects.size(); r++)
        {
            if (rects[r].height >= m_hp && rects[r].width >= m_wp)
            {
                cv::Mat subject (md.getFrame(f), rects[r]); // Get a roi in frame defined by the rectangle.
                cv::Mat maskedSubject;
                subject.copyTo(maskedSubject, md.getPredictedMask(f,r));
                subject.release();
                
                GridMat g (maskedSubject, m_hp, m_wp);
                gmasks.push_back( g );
                
                grects.push_back(rects[r]);
                tagsAux.push_back(tags[r]);
            }
        }
    }
    
    cv::Mat tmp (tagsAux.size(), 1, cv::DataType<int>::type, tagsAux.data());
    tmp.copyTo(gtags);
}