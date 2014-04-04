function post = EigModel_posterior( data, models, priors )

% post = EigModel_posterior( data, models, priors )
% Compute the posterior probability for each datum (each a column vector)
% in the matrix data, given an array of EigModel in models, and prior for
% each EigModel in priors.
% We assume sum(priors) == 1 and length(models) == length(priors).
% pre:
%    data is the data for which the posterior is to be computed
%    models is an erray of EigModel
%    priors is a arrar of priors for each EigModel
% post:
%   post is and array of posterior probabilities, one for each datum.

models

post = zeros( length(models), size(data,2) );
for i = 1:length(models)
  % compute Mahalnaobis distance, even in null space if necessary
  post(i,:) =  EigModel_likelihood( data, models{i},  'MogPen' );
  % compute likelihoods
  %post(i,:) = exp( -0.5*post(i,:) );
  %post(i,:) = post(i,:) / ( sqrt(prod(models{i}.val))*(2*pi)^(length(models{i}.val)/2) );
  % weight likelihoods by prior
  post(i,:) = post(i,:)*priors(i);
end

% ensure sum of posterioir probailities for each point is 1
post = post ./ repmat( sum( post, 1)+eps ,length(models),1);

return;
