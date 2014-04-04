function like = EigModel_likelihood( data, model , opt )

% like = EigModel_likelihood( data, model )
% Compute the likelihood for the data given the EigModel
% pre:
%    data - the data set matrix  in which each datum is a colum
%    model - the EigModel
%    opt - if set to 'MogPen' this takes in acount any residue in the null space.
% post:
%   like - the likelihood of each datum as measured against the model


if model.N == 0
  % an empty EigeModel
  like = NaN*ones( 1, size(data,2) );
  return;
end

flag = 0; % any null space don't matter
if nargin == 3
  if strcmp(opt,'MogPen')
    if size( model.vct,1) ~= size( model.vct, 2 );
      % take null space into account when estimating likelihood
      flag = 1;
    end
  end
end


if flag == 0
  like =  exp( -0.5*EigModel_Mahalanobis( data, model ) ) / sqrt( prod(model.val) * (2*pi)^(size(model.val,1)) );
else
  % take null space into account
  D = size( model.vct,1) - size( model.vct,2 ); % dimension of null space
  like =  exp( -0.5*EigModel_Mahalanobis( data, model, 'MogPen') ); % estimate un-normalised likelihood
  vol = prod( model.val ); % volume of Eigenmodel in Eigenspace
  if model.residue > 0
    vol= vol*(model.residue/D)^D; % scale by volume in null space
  end
  if vol == 0
    vol = realmin;
  end
  like = like / sqrt( vol*(2*pi)^(size(model.vct,1)) );
end
  

return;



% compute the normalisation factor
k = prod(model.val) * (2*pi)^(size(model.vct,1));
% adjust this factor, if necessary
d = size(model.vct,1) - size(model.vct,2); % d is the number of missing dimensions
if flag & d > 0
  kk = model.residue / d; % the average spectral energy per dimension
  k = kk^d*k;
  if k < eps^2 % if the scale factor is very small indeed, call it zero
    k = kk* eps * 2 * pi;
  end
  if k < eps^2
    k = eps^2;
  end
end

like = like / sqrt(k);

return;
