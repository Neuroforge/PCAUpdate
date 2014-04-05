cv::Mat EigModel_add( cv::Mat m1Vec,cv::Mat m1Vals, cv::m1Mean, cv::Mat m2Vec,cv::Mat m2Vals, cv::Mat m2Mean)
{

	//Daniel's notes on this code
	//1. Origin is used to describe the mean for the vector spaces.
	//2. We need the eigenvalues for m1 and m2 also, denoted below as m1.val //m2.val 

	// Peter Hall's notes.
	// m3 = EigModel_add( m1, m2, method, param  )
	//
	// Add eigenmodels m1 and m2 to produce eigenmodel m3
	// Keep eigenvectors using method specified by retmethod,
	// to an extent specified by retparam.
	// pre:
	//  m1 is an EigModel
	//  m2 is an EigModel
	//  retmethod is an optional string, must be one of:
	//    'keepN' - to keep N of the eigenvectors
	//    'keepf' - to keep a // of the eigenenergy
	//    'keept' - to keep eigenvalues above a threshold
	//    'keepr' - to keep a faction of the eigenvectors
	//  retparam is compulsory if retmethod is defined, prohibited otherwise.
	//    It is a numerical value whose meaning depend on retmethod.
	// post:
	//  m3 is an eigenmodel:
	// Notes:
	//  For an EigModel m
	//    m.org is the mean of the observations.
	//    m.vct is a (n x p) matrix of eigenvectors (columns)
	//	   p is decided by the <retmethod> above.
	//    m.val is a (p x 1)  matrix (column vector) of eigenavlues.
	//    m.N is the number of observations input to make the model.
	//	 The ith eigenvalue and eigenvecvtor correspond, and the e-values
	//	 are ordered by decending value.
	//	 Hence, the ordering of the eigenvectors may different from that
	//	 of any support vectors input, and there is no guarantee that
	//	 the eigenvectors form a right-handed set.
	// See also:
	//  EigModel_make
	//  EigModel_rank


	//WHAT ARE THE RESIDUES IN THE CONTEXT OF THE MODEL???

	// Compute the new number of observations
	int N = m1Vec.rows;
	int M = m1Vec.rows;
	int P = M*N

	// The body of the function follows....

	// Compute the new origin
	m3Mean = (N*m1Mean + M*m2Mean)/P;

	// Store the vector between origins for later use
	cv::Mat dorg = cv::subtract(m1Mean, m2Mean);

	// Compute a new spanning basis
	cv::Mat G = m1Vec.transpose() * m2Vec;
	cv::Mat H = m2Vec - m1Vec * G;
	cv::Mat g = m1Vec.transpose() * dorg;
	cv::Mat h = dorg - m1Vals * g; // residue wrt X

	if ~isempty( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] )
	{
	  nu = orth( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] );
	  // make sure - errors can occur if the deflated dimensionof X is less than that of Y
	  H = m1Vec.transpose() * nu;
	  
	  
	  //WHAT DOES THE PERIOD AFTER H MEAN???
	  nu = nu( :, sum( H.*H, 1 ) < eps );
	}
	else
	{
	  nu = cv::Mat::zeros( m1Mean,1), 0 );
	}

	// The residue in each gives the energy in the residue space.
	// The size of the space may have changed - in fact may dissapear.
	// so here compute the residue per "direction"...
	resn1 = m1Vec.rows-m1Vec.cols;
	if (resn1 > 0)
	{
	  rpern1 = m1Vec.residue / resn1;
	}
	else
	{
	  rpern1 = 0;
	}

	int resn2 = m2Vec.rows-m2Vec.cols;
	if resn2 > 0
	{
	  rpern2 = m2Vec.residue / resn2;
	}
	else
	{
	  rpern2 = 0;
	}

	// Compute the intermediate matrix as the sum of three matrices, A1,A2,A3...
	// A term for the correlation of m1, use reside for error correction

	[n m] = nu.size();
	A1 = (N/P)*cv::diag( [m1Vals.transpose(),  rpern1*cv::Mat::ones(1,m, CV_64FC1)]);

	// A term for the correlation of m2; project m2.vct onto the new basis
	// use residue for error correction
	Gamma = nu.transpose() * m2Vec;
	D = G*dcv::iag(m2Vals);
	E = Gamma*diag(m2Vals);
	
	
	//WHAT IS ALL OF THIS CRAZY BRACKET SCHTICK??? [;...]????
	A2 = (m2Vec.N/m3Vec.N)*[ D*G.transpose() D*Gamma.transpose(); ...
								  E*G.transpose() E*Gamma.transpose()] + rpern2*eye(size(A1,1));

	// A term for the difference between means
	gamma = nu.transpose() * dorg;
	A3 = (N*M)/(P^2)*[g*g.transpose() g*gamma.transpose(); ...
							   gamma*g.transpose() gamma*gamma.transpose()];

	// guard against rounding errors forcing imaginary values!
	A = (A1+A2+A3);
	A = (A+A')/2; 

	cv::PCA pca = cv::PCA(A, NULL, cv::CV_PCA_AS_ROW);

	// now can compute...
	[m3Vec m3Val] = eig( A ); // the eigen-solution
	m3Vec = [m1 nu]* m3; // rotate the basis set into place - can fail for v.high dim data
	m3Vals = cv::diag(pca.eignevalues);             // keep only the diagonal

	////NEED TO REVIEW PAPER FOR THIS ONE/////


	//DOESNT OPENCV DO THIS FOR US ALREADY?
	if nargin == 4 // Deflate the EigModel
	{
	  n = Emodel_rank( m3.val, method, param );
	  m3Val = m3Val( 1:n );
	  m3Vec = m3Vec( :, 1:n );
	}
	else // make sure every eigenvalue is >= 0
	{
	  n = Emodel_rank( m3Vals, 'keept', eps ); // max actual rank
	  m3Vals = m3Vals( 1:n );
	  m3Vec = m3Vec( :, 1:n );
	}

	resn3 = m3Vec.rows - m3Vec.cols;
	// the add the residues per direction, and scale by the number of residue
	// directions in the result.
	m3.residue = resn3*( rpern1 + rpern2 );
}
int main(int argc, char* argv[])
{
}
