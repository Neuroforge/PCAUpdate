#ifndef PCAMERGE_H
#define PCAMERGE_H

// Must change header to only the required files like core,etc.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

class PCAMerge
{
	cv::PCA m1;
	cv::PCA m2;
	
	cv::PCA m3;
	int N;
	int M;

	public:
	
	cv::Mat eigenVals;
	cv::Mat eigenVecs;
	cv::Mat mean;
	cv::Mat nObs;

	bool addModel1( cv::PCA pcaM1, int n1);
	bool addModel2( cv::PCA pcaM2, int n2);

	void computeAdd();
	
}

cv::Mat orth( cv::Mat vecs);
#endif
