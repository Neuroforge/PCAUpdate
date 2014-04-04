function emodel = EigModel_robust_make( z )

w = ones( 1, size(z,2) );
w = w / sum( w );
emodel = EigModelW_make( z, w );

tol = 1-1E-6;
for i = 1:6
  eold = emodel;
  w = (1-erf( EigModel_Mahalanobis( z, eold)/3 - 3 ))/2; % a new set of weights
  w = w / sum( w );
  emodel = EigModelW_make( z, w ); % a new model
  d = EigModel_Chernoff( eold, emodel ); % some distancte to old model
  if d > tol
    break;
  end
end

return;
