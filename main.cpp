cv::Mat EigModel_add( cv::Mat m1, cv::Mat m2, std::string method, std::string param)
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


	// Compute the new number of observations
	int N = m1.rows;
	int M = m1.rows;
	int P = M*N

	// The body of the function follows....

	// Compute the new origin
	m3Mean = (N*m1Mean + M*m2Mean)/P;

	// Store the vector between origins for later use
	cv::Mat dorg = cv::subtract(m1Mean, m2Mean);

	// Compute a new spanning basis
	cv::Mat G = m1.transpose() * m2;
	cv::Mat H = m2 - m1 * G;
	cv::Mat g = m1.transpose() * dorg;
	cv::Mat h = dorg - m1.vct * g; // residue wrt X

	if ~isempty( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] )
	{
	  nu = orth( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] );
	  // make sure - errors can occur if the deflated dimensionof X is less than that of Y
	  H = m1.transpose() * nu;
	  nu = nu( :, sum( H.*H, 1 ) < eps );
	}
	else
	{
	  nu = cv::Mat::zeros( m1Mean,1), 0 );
	}

	// The residue in each gives the energy in the residue space.
	// The size of the space may have changed - in fact may dissapear.
	// so here compute the residue per "direction"...
	resn1 = m1.rows-m1.cols;
	if (resn1 > 0)
	{
	  rpern1 = m1.residue / resn1;
	}
	else
	{
	  rpern1 = 0;
	}

	int resn2 = m2.rows-m2.cols;
	if resn2 > 0
	{
	  rpern2 = m2.residue / resn2;
	}
	else
	{
	  rpern2 = 0;
	}

	// Compute the intermediate matrix as the sum of three matrices, A1,A2,A3...
	// A term for the correlation of m1, use reside for error correction

	[n m] = nu.size();
	A1 = (N/P)*cv::diag( [m1.val',  rpern1*cv::Mat::ones(1,m, CV_64FC1)]);

	// A term for the correlation of m2; project m2.vct onto the new basis
	// use residue for error correction
	Gamma = nu.transpose() * m2;
	D = G*dcv::iag(m2.val);
	E = Gamma*diag(m2.val);
	A2 = (m2.N/m3.N)*[ D*G' D*Gamma'; ...
								  E*G' E*Gamma'] + rpern2*eye(size(A1,1));

	// A term for the difference between means
	gamma = nu' * dorg;
	A3 = (N*M)/(P^2)*[g*g.transpose() g*gamma.transpose(); ...
							   gamma*g.transpose() gamma*gamma.transpose()];

	// guard against rounding errors forcing imaginary values!
	A = (A1+A2+A3);
	A = (A+A')/2; 

	cv::PCA pca = cv::PCA(A, NULL, cv::CV_PCA_AS_ROW);

	// now can compute...
	[m3.vct m3.val] = eig( A ); // the eigen-solution
	m3 = [m1 nu]* m3; // rotate the basis set into place - can fail for v.high dim data
	m3val = cv::diag(pca.eignevalues);             // keep only the diagonal

	////NEED TO REVIEW PAPER FOR THIS ONE/////

	if nargin == 4 // Deflate the EigModel
	  n = Emodel_rank( m3.val, method, param );
	  m3.val = m3.val( 1:n );
	  m3.vct = m3.vct( :, 1:n );
	else // make sure every eigenvalue is >= 0
	  n = Emodel_rank( m3.val, 'keept', eps ); // max actual rank
	  m3.val = m3.val( 1:n );
	  m3.vct = m3.vct( :, 1:n );
	end


	resn3 = size( m3.vct,1 ) - size( m3.vct,2 );
	// the add the residues per direction, and scale by the number of residue
	// directions in the result.
	m3.residue = resn3*( rpern1 + rpern2 );
}
int main(int argc, char* argv[])
{
}