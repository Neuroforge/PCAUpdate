#include "PCAMerge.h"

cv::Mat orth( cv::Mat vecs )
{
	//Modified GS Orthonormalization Process
	cv::Mat temp;
	cv::Mat cveci, cvecj;
	vecs.copyTo( temp );

	for( int i=0;i < temp.cols; i++ )
	{
		cveci = temp.col(i);
		normalize( cveci, cveci );
		for( int j=i+1; j<temp.cols; j++)
		{
			cvecj = temp.col(j);
			cvecj = cvecj - ( ( (cvecj.dot(cveci)) / (cveci.dot(cveci)) ) * cveci );
		}
	}

	//Check if every vector is orthonormal to each other
	std::cout << temp.t() * temp << std::endl;

	return temp;
}

bool PCAMerge::addModel1( cv::PCA pcaM1, int n1)
{
	//Sets the model1. Must add additional error corrections
	if( n1 == 0 || pcaM1.eigenvalues.empty() || pcaM1.eigenvectors.empty() )
	{
		m1.eigenvalues.data = NULL;
		m1.eigenvectors.data = NULL;
		
		m1.mean.data = NULL;
		N = 0;

		return false;
	}

	else
	{
		//Does it copy data or create reference ????
		m1 = pcaM1;
		N = n1;
		return true;
	}

}


bool PCAMerge::addModel2( cv::PCA pcaM2, int n2)
{
	// Similar to above function. Sets the second model to be combined.
	if( n2 == 0 || pcaM2.eigenvalues.empty() || pcaM2.eigenvectors.empty() )
	{
		m2.eigenvalues.data = NULL;
		m2.eigenvectors.data = NULL;
		m2.mean.data = NULL;
		M = 0;

		return false;
	}

	else
	{
		//Does it copy data or create reference???
		m2 = pcaM2;
		M = n2;
		return true;
	}

}
//TODO Must make sure we always use 32F or 64F.
void PCAMerge::computeAdd()
{
	// Add checks for errors, Exceptions
	// TODO : Add checks against model, number of observations. Create eigenVals and eigenVecs accordingly. Refer Matlab code for possible checks to be implemented
	//
	// From here, going forward assuming no other errors possible
	//
	
	float eps = 1e-3;
	// New number of Observations
	nObs = N + M;
	// TODO add check for nObs being zero
	
	// New model's mean.
	mean = ( (N * m1.mean) + (M * m2.mean) ) / nObs;

	// Vector joining the centres
	cv::Mat dorg = m1.mean - m2.mean;

	// Note:
	// dorg            : n x 1
	// m1.eigenvectors : n x p
	// m2.eigenvectors : n x q
	// G			   : p x q
	// H			   : n x q
	// g 			   : p x 1
	// h               : n x 1
	
	//Note Eigenvectors from cv::PCA are stored as rows. We need to transpose them.
	
	m1.eigenvectors = m1.eigenvectors.t();
	m2.eigenvectors = m2.eigenvectors.t();

	//New Basis
	cv::Mat G = m1.eigenvectors.t() * m2.eigenvectors; 
	cv::Mat H = m2.eigenvectors - ( m1.eigenvectors * G );// H is orthogonal to m1.eigenvectors
	
	cv::Mat g = m1.eigenvectors.t() * dorg;
	cv::Mat h = dorg - (m1.eigenvectors * g); // h is orthogonal to dorg

	//Some vectors in H can be zero vectors. Must be removed
	cv::Mat sumH = cv::Mat::zeros( 1, H.cols, CV_64FC1 );
	cv::reduce( H.mul(H), sumH, 0, cv::REDUCE_SUM );
	// Even h can be a zero vector. Must not be used if so
	double sumh = 0;
	sumh = h.dot(h);
	//
	// Get indices of sumH > eps. use it to construct vector nu
	cv::Mat newH;
	for( int i=0; i < sumH.cols; i++ )
	{
		if( sumH.at<double>(i) > eps )
			newH.push_back( H.col(i).t() );
	}

	if (sumh > eps)
		newH.push_back( h.t() );

	newH = newH.t();
	// Dimension of newH must be n x t
	std::cout << newH.size() << std::endl;
	
	//TODO : Implement Gram Schmidt Orthonormalization. DONE
	cv::Mat nu = orth( newH );

	//TODO : Forgetting about residues at the moment.
	//Residues are the eigenvalues which were not used in the model m1 / m2.
	//The following was used in matlab for including residues
	//presn1 = size( m1.vct, 1) - size(m1.vct,2 );
	/* if resn1 > 0
  		rpern1 = m1.residue / resn1;
	   else
 		rpern1 = 0;
	   end

		resn2 = size( m2.vct, 1) - size(m2.vct,2 );
		if resn2 > 0
  			rpern2 = m2.residue / resn2;
		else
  			rpern2 = 0;
		end
	*/

	//First part of the matrix in equation (20) in paper - Correlation of m1
	//
	int n,p,t,q;

	n = m1.eigenvectors.rows; // = m2.eigenvectors.rows
	t = nu.cols; //
	p = m1.eigenvalues.rows;
	q = m2.eigenvalues.rows;

	cv::Mat tempeval = cv::Mat::zeros( (p + t) , 1, m1.eigenvalues.type() );
	m1.eigenvalues.copyTo( tempeval.rowRange(0,m1.eigenvalues.rows) );

	cv::Mat A1 = ( N / nObs ) * cv::Mat::diag(tempeval);

	// Correlation of m2
	cv::Mat Gamma = nu.t() * m2.eigenvectors;
	cv::Mat D     = G * cv::Mat::diag( m2.eigenvalues );
	cv::Mat E     = Gamma * cv::Mat::diag( m2.eigenvalues);

	cv::Mat A2 = cv::Mat::zeros( A1.size(), A1.type() );

	A2( cv::Range(0,p), cv::Range(0, p) ) = D * G.t();
	A2( cv::Range(0,p), cv::Range(p, A1.cols) ) = D * Gamma.t();
	A2( cv::Range(p, A1.rows), cv::Range(0,p) ) = E * G.t();
	A2( cv::Range(p, A1.rows), cv::Range(p, A1.cols) ) = E * Gamma.t();

	A2 = A2 * ( M / nObs );

	//Third Part : term for diff between means
	cv::Mat gamma = nu.t() * dorg;
	cv::Mat A3 = cv::Mat( A1.size(), A1.type() );
	
	A3( cv::Range(0,p), cv::Range(0,p) ) = g * g.t();
	A3( cv::Range(0,p), cv::Range(p, A1.cols) ) = g * gamma.t(); 
	A3( cv::Range(p, A1.rows), cv::Range(0,p) ) = gamma * g.t(); 
	A3( cv::Range(p, A1.rows), cv::Range(p, A1.cols) ) = gamma * gamma.t();

	A3 = ( N * M / (nObs*nObs) ) * A3;

	// Guard against rounding errors
	cv::Mat A = A1 + A2 + A3;
	A = ( A + A.t() ) / 2.0;

	m3 = cv::PCA( A, cv::noArray(), cv::PCA::DATA_AS_ROW );

	eigenVals = m3.eigenvalues;
	m3.eigenvectors = m3.eigenvectors.t();

	cv::Mat m3Temp = cv::Mat::zeros( n, A.cols, A.type() );

	m3Temp( cv::Range::all(), cv::Range(0,p)) = m1.eigenvectors;
	m3Temp( cv::Range::all(), cv::Range(p, A.cols)) = nu;

	eigenVecs = m3Temp * m3.eigenvectors;

	//Look at how many eigenvalues must be returned. Call that function as required.
}


int main( int argc, char** argv)
{
	cv::Mat A = ( cv::Mat_<double>(3,3) << 0.5,0,1,0,1,0,1,0,0);
	cv::Mat B = orth(A);

	return 0;
}
