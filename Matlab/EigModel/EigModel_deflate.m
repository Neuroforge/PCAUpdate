function m2 = EigModel_deflate( m1, method, param )

% m2 = EigModel_deflate( m1, method, param )
% Deflate the EigModel m1 using the specified method, to yield m2.
% pre:
%	val is a (N x 1) matrix of eigenvalues, sorted in descending order.
%	method is a string: one of
%		'keepN' - to keep a fixed number of the eignvectors
%		'keepf' - to keep a specific fraction of the eigenenery;
%		'keept' - to keep all eigenvectors above a threshold
%		'keepr' - to keep a specific fraction of the number of evals.
%	param is a parameter, assumed to have a value appropriate to
%		the method chosen.
% post:

m2.N = m1.N;
m2.org = m1.org;
n = Emodel_rank(  [m1.val], method, param );
dummy = sum( m1.val ); % total Eigenenergy
m2.val = m1.val( 1:n );
m2.vct = m1.vct( :, 1:n );
m2.residue =  dummy - sum( m2.val ) + m1.residue;

return;
