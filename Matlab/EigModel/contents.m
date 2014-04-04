% A library for working with eigenmodels of multivariate data
%
% An overview
% -----------
% An eigenmodel represents a statistical description of multivariate data,
% which should be stored in a matrix. Each column of the matrix is a data point,
% data points need not be unique. The eigenmodel description comprises four
% components: (1) the number of data points; (2) the 1st moment (average)
% of the data points - their centre, or origin; (3) a set of basis vectors,
% that are a best linear fit to the data, through their origin; (4) a set of
% non-negative, real scalars. There are as many scalars as vectors, and they
% mutually correspond by index. Each scalar is the standard deviations of the
% points projected onto the corresponding vector. The vectors and scalars are,
% in fact, the eigenvectors and eigenvalues of the correlation matrix of the
% mean-shifted data; the covariance matrix is scaled by the reciprocal of
% the number of points.
%
% Eigenmodels contain only as many eigenvectors and eigenvalues as the rank
% of the (scaled) correlation matrix. This rank is at most the smallest of the
% number of points or number of dimensions. Any linear correlation between
% data yields a lower rank. Small an eigenvalues imply near-linear correlations
% in a direction perpendicular to the eigenvector; such eigenvectors and values
% may be discarded - a process known as deflation.
%
% The library
% -----------
% An EigModel is a structured type that specifies an eigenmodel.
% The components of an EigModel are:
%  N   - the number of data points
%  org - their origin
%  vct - the eigenvectors
%  val - the eigenvalues
% Thus if m in an EigModel, then m.N, m.org, m.vct, m.val
% are valid MATLAB contructs.
% Functions currently in the library are:
%   isEigModel
%   EigModel_make
%   EigModel_add
%   EigModel_rank
%   EigModel_deflate
%   EigModel_project
%   EigModel_unproject
%   EigModel_Mahalanobis
%   EigModel_likelihood
%   EigModel_posterior
%   EigModel2D_draw
%   EigModel_Chernoff (requires EigModel_Chernoff_fmin)
% Functions I may include at some future time are:
%   EigModel_subtract


