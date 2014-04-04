function n = Emodel_rank( val, method, param )

% n = Emodel_rank( val,method,param)
%
% Compute number of eignvectors/values to keep,
% using some specified method.
%
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
%	n - the number to keep
% notes:
%  For keepN method, the n returned is the minimum of param and the number
%  of eigenvalues.

if isempty(val)
  n = 0;
  return;
end

if strcmp(method,'keepN')
  n = min(param,size(val,1));
elseif strcmp(method,'keepf')
  s = sum(abs(val)).*param; % total power (values are squared already)
  ss = abs(val(1));
  m = size(val,1); % number of eigenvalues
  n = 2;
  while (ss <= s) & (n <= m)
    ss = ss + abs(val(n));
    n = n+1;
  end
  n = n-1; % back up one place
elseif strcmp(method,'keept')
  n = sum(val > param);
elseif strcmp(method,'keepr')
  n = min( size(val,1), round(size(val,1)*param) );
end

return;
