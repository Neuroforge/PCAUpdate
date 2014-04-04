function emodel = EigModelW_make( data,weight )

% emodel = EigModelW_make( data, weight )
% Make an eigenmodel given weights on each data point

emodel.N = size(data,2);
emodel.w0 = sum( weight );

wdata = data.*repmat( weight / emodel.w0, size(data,1), 1);

emodel.org = sum( wdata, 2 );

C = wdata * data' - emodel.org * emodel.org';

[u,s] = eig( C );
s = diag(s);

[dummy,i] = sort( s );
i = flipud( i );

emodel.vct= u(:,i);
emodel.val = s(i);


return;




