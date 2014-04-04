function y = EigModel_whiten( x, e )
%% EigModel_whiten
% perform a whitening transform on x, given eigenmodel e.

y = e.vct'*(x - repmat( e.org, 1, size(x,2) )) ./ repmat(sqrt(e.val),1,size(x,2));

end

