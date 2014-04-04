function k = EigModel_Chernoff_fmin( beta, cx, cy, d, ncx, ncy)

% k = EigModel_Chernoff_fmin( beta, cx, cy, d, ncx, ncy )
% Return the "Chernoff distance" at parameter beta. This distance
% has value 1 when two distributions are equal, and 0 when they nowhere
% intersect: the Chernoff distance is the probabilty that the two distributions
% are the same.


beta1 = 1-beta;
S = beta*cx + beta1*cy;
% if rank(S) < size(S,1)
%   fprintf('deg S in EigMOdel_Chernodd_fmin, hit a key to continue or ^c to break\n');
%   pause;
% end
ns = abs(det(S));

k = exp( -(0.5*beta*beta1)*(d' * inv(S) * d) ) * sqrt ( (ncx^beta * ncy^beta1) / ns );
     
return;