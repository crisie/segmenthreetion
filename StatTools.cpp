//
//  StatTools.cpp
//  segmenthreetion
//
//  Created by Albert Clapés on 17/02/14.
//
//

#include "StatTools.h"

#include <math.h>

// Instantiation of template member functions
// -----------------------------------------------------------------------------
template void variate<int>(vector<vector<int > > list, vector<vector<int > >& variations);
template void variate<float>(vector<vector<float > > list, vector<vector<float > >& variations);
template void variate<double>(vector<vector<double > > list, vector<vector<double > >& variations);

template void expandParameters<int>(vector<vector<int> > params, vector<vector<int> >& expandedParams);
template void expandParameters<float>(vector<vector<float> > params, vector<vector<float> >& expandedParams);
template void expandParameters<double>(vector<vector<double> > params, vector<vector<double> >& expandedParams);

template void expandParameters<int>(vector<vector<int> > params, int ncells, vector<vector<int> >& expandedParams);
template void expandParameters<float>(vector<vector<float> > params, int ncells, vector<vector<float> >& expandedParams);
template void expandParameters<double>(vector<vector<double> > params, int ncells, vector<vector<double> >& expandedParams);

template void selectParameterCombination<int>(vector<vector<int> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);
template void selectParameterCombination<float>(vector<vector<float> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);
template void selectParameterCombination<double>(vector<vector<double> > expandedParams, int hp, int wp, int nparams, int idx, vector<cv::Mat>& selectedParams);

template void selectBestParameterCombination<int>(vector<vector<int> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
template void selectBestParameterCombination<float>(vector<vector<float> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);
template void selectBestParameterCombination<double>(vector<vector<double> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams);

// -----------------------------------------------------------------------------


/**
 * Builds an histogram of the values contained in a vector (or matrix)
 */
void histogram(cv::Mat mat, int nbins, cv::Mat & hist)
{
    double minval, maxval;
    cv::minMaxIdx(mat, &minval, &maxval);
    
    if (nbins > (maxval-minval+1))
        return;
    
    // Create an histogram for the cell region of blurred intensity values
    int histSize[] = { (int) nbins };
    int channels[] = { 0 }; // 1 channel, number 0
    float tranges[] = { (float)minval, (float)maxval }; // thermal intensity values range: [0, 256)
    const float* ranges[] = { tranges };
    
    cv::calcHist(&mat, 1, channels, cv::noArray(), hist, 1, histSize, ranges, true, false);
}

/**
 * Create a column vector containing the numbers in the interval [a,b] shuffled randomly
 */
cv::Mat shuffledVector(int a, int b, cv::RNG randGen)
{
    cv::Mat vec (b-a+1, 1, cv::DataType<int>::type);
    for (int i = a; i <= b; i++)
    {
        vec.at<int>(i-a, 0) = i;
    }
    
    randShuffle(vec, 1, &randGen);
    
    return vec;
}

/**
 * Create a column vector containing the numbers in the interval [0,n) shuffled randomly
 */
cv::Mat shuffledVector(int n, cv::RNG randGen)
{
    return shuffledVector(0, n-1, randGen);
}

/**
 * Create a vector of labels representing the k folds of n elements
 */
void cvpartition(int n, int k, int seed, cv::Mat& partitions)
{
    int foldElems = std::floorf(n/k);
    int extraElems = n - (k * std::floorf(n/k));

    cv::Mat indices = shuffledVector(n, cv::RNG(seed));
    
    partitions.release();
    partitions.create(indices.rows, indices.cols, cv::DataType<int>::type);

    unsigned int c = 0;
    unsigned int i, j;
    for (i = 0; i < k; i++)
    {
        int ifoldElems = (i < extraElems) ? (foldElems + 1) : foldElems;
        for (j = 0; j < ifoldElems; j++)
        {
            partitions.at<int>(indices.at<int>(c+j)) = i;
        }
        
        c += ifoldElems;
    }
}


void cvpartition(GridMat gclasses, int k, int seed, GridMat& gpartitions)
{
    gpartitions.release();
    gpartitions.create(gclasses.crows(), gclasses.ccols());
    for (int i = 0; i < gclasses.crows(); i++) for (int j = 0; j < gclasses.ccols(); j++)
    {
        cvpartition(gclasses.at(i,j), k, seed, gpartitions.at(i,j));
    }
}

/**
 * Create a vector of labels representing the k folds of n elements (stratified)
 */
void cvpartition(cv::Mat classes, int k, int seed, cv::Mat& partitions)
{
    double minVal, maxVal; // labels do not need to be between [0, #classes - 1]
    cv::minMaxIdx(classes, &minVal, &maxVal); // but, suppose continuous numeration of labels
    
    // separate the indices in a different vector for each class

    std::vector<std::vector<int> > classesIndices(maxVal - minVal + 1);
    int m = (classes.rows > 1) ? classes.rows : classes.cols;
    for (int i = 0; i < m; i++)
    {
        int l = (classes.rows > 1) ? classes.at<int>(i,0) : classes.at<int>(0,i);
        classesIndices[l - minVal].push_back(i);
    }
    
    // perform partitions separately in the classes' indices vectors
    // and then merge the separate partitions into one
    
    std::vector<cv::Mat> classesPartitions(maxVal - minVal + 1);
    
    partitions.release();
    partitions.create(classes.rows, classes.cols, cv::DataType<int>::type);
    
    for (int i = 0; i < classesIndices.size(); i++)
    {
        cvpartition(classesIndices[i].size(), k, seed, classesPartitions[i]);
        for (int j = 0; j < classesIndices[i].size(); j++)
        {
            (partitions.rows > 1) ?
                partitions.at<int>(classesIndices[i][j], 0) = classesPartitions[i].at<int>(j,0) :
                partitions.at<int>(0, classesIndices[i][j]) = classesPartitions[i].at<int>(j,0);
        }
    }
}


/**
 * Mathematical function approximating a Gaussian function
 */
double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    
    return 0.5*(1.0 + sign*y);
}

/**
 * Converts Mat to Vector
 */
void matToVector(cv::Mat image, vector<int> & values)
{
    cv::Mat_<uchar>::iterator it_start = image.begin<uchar>();
    cv::Mat_<uchar>::iterator it_end = image.end<uchar>();
    
    for(; it_start != it_end; ++it_start) {
        values.push_back(*it_start);
    }
}

/**
 * Sort a vector by unique values
 */
void uniqueSortValues(vector<int> & values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

/**
 * Find unique values of a Mat and returns them sorted
 */
void findUniqueValues(cv::Mat image, vector<int> & values) {
    matToVector(image, values);
    uniqueSortValues(values);
}

/**
 * Find unique values of a vector and returns them sorted
 */
void findUniqueValues(vector<int> v, vector<int> & values) {
    uniqueSortValues(v);
    values = v;
}

template<typename T>
void variate(vector<vector<T > > list, vector<vector<T > >& variations)
{
    vector<T> v(list.size()); // empty
    _variate(list, 0, v, variations);
}

template<typename T>
void _variate(vector<vector<T > > list, int idx, vector<T> v, vector<vector<T > >& variations)
{
    if (idx == list.size())
    {
        return;
    }
    else
    {
        for (int i = 0; i < list[idx].size(); i++)
        {
            v[idx] = list[idx][i];
            _variate(list, idx+1, v, variations);
            if (idx == list.size() - 1)
            {
                variations.push_back(v);
                //v.erase(v.begin()+idx);
            }
        }
    }
    
}


template<typename T>
void expandParameters(vector<vector<T> > params, vector<vector<T> >& expandedParams)
{
    variate(params, expandedParams);
}


template<typename T>
void expandParameters(vector<vector<T> > params, int ncells, vector<vector<T> >& gridExpandedParams)
{
    vector<vector<T> > cellExpandedParams;
    variate(params, cellExpandedParams);
    
    // Create and expand a list of indices, used to index the cellExpandedParams
    
    vector<int> indices(cellExpandedParams.size());
    
    for (int i = 0; i < cellExpandedParams.size(); i++)
        indices[i] = i;
    
    vector<vector<int> > listsOfIndices(ncells);
    for (int i = 0; i < ncells; i++)
        listsOfIndices[i] = indices;
    
    vector<vector<int> > expandedIndices;
    variate(listsOfIndices, expandedIndices);
    
    //    // debug
    //    for (int i = 0; i < expandedIndices.size(); i++)
    //    {
    //        cv::Mat m (expandedIndices[i].size(), 1, cv::DataType<int>::type, expandedIndices[i].data());
    //        cout << m << endl;
    //    }
    //    //
    
    // Create the grid's combinations' list of parameters
    
    gridExpandedParams.clear();
    gridExpandedParams.resize(expandedIndices.size());
    
    for (int i = 0; i < expandedIndices.size(); i++)
    {
        for (int j = 0; j < expandedIndices[i].size(); j++)
        {
            vector<T> combination = cellExpandedParams[expandedIndices[i][j]];
            for (int k = 0; k < params.size(); k++)
            {
                gridExpandedParams[i].push_back(combination[k]);
            }
        }
        //        // debug
        //        cv::Mat m (gridExpandedParams[i].size(), 1, cv::DataType<T>::type, gridExpandedParams[i].data());
        //        cout << m << endl;
    }
}


template<typename T>
void selectParameterCombination(vector<vector<T> > expandedParams, int hp, int wp,
                                int nparams, int idx, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
    
    vector<T> lineParams = expandedParams[idx];
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
    //    for (int k = 0; k < nparams; k++)
    //        cout << selectedParams[k] << endl;
}


template<typename T>
void selectBestParameterCombination(vector<vector<T> > expandedParams, int hp, int wp, int nparams, GridMat goodnesses, vector<cv::Mat>& selectedParams)
{
    selectedParams.clear();
    
    for (int k = 0; k < nparams; k++)
        selectedParams.push_back(cv::Mat(hp, wp, cv::DataType<T>::type));
    
    GridMat gargmax;
    goodnesses.argmax<T>(gargmax);
    
    for (int i = 0; i < hp; i++) for (int j = 0; j < wp; j++)
    {
        int rowIdx = gargmax.at<T>(i,j,0,0); // maxrow index
        
        vector<T> lineParams = expandedParams[rowIdx];
        
        int l = i * wp + j;
        for (int k = 0; k < nparams; k++)
        {
            selectedParams[k].at<T>(i,j) = lineParams[l * nparams + k];
        }
    }
    
    // debug
    //    for (int k = 0; k < nparams; k++)
    //        cout << selectedParams[k] << endl;
}

float accuracy(cv::Mat actuals, cv::Mat predictions)
{
    cv::Mat objects  = (actuals == 0);
    cv::Mat subjects = (actuals == 1);
    cv::Mat hits = (actuals == predictions);
    
    int nobj = cv::sum(objects).val[0];
    int nsbj = cv::sum(subjects).val[0];
    
    int objHits = cv::sum(objects & hits).val[0];
    int sbjHits = cv::sum(subjects & hits).val[0];
    
    return (float(objHits)/nobj + float(sbjHits)/nsbj) / 2;
}

void accuracy(GridMat actuals, GridMat predictions, cv::Mat& accuracies)
{
    accuracies.create(predictions.crows(), predictions.ccols(), cv::DataType<float>::type);
    
    for (int i = 0; i < predictions.crows(); i++) for (int j = 0; j < predictions.ccols(); j++)
    {
        cv::Mat objects  = (actuals.at(i,j) == 0);
        cv::Mat subjects = (actuals.at(i,j) == 1);
        cv::Mat hits = (actuals.at(i,j) == predictions.at(i,j));
        
        int nobj = cv::sum(objects).val[0];
        int nsbj = cv::sum(subjects).val[0];
        
        int objHits = cv::sum(objects & hits).val[0];
        int sbjHits = cv::sum(subjects & hits).val[0];
        
        accuracies.at<float>(i,j) = (float(objHits)/nobj + float(sbjHits)/nsbj) / 2;
    }
}
