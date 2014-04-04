% EigModel_make_test.m
clear all;
close all;

% Test for empty data
%
n = 10;
N = 0;  % no observations
obs0 = rand(n,N);
m0 = EigModel_make( obs0 );

fprintf('zero model\n');
m0.N
m0.org
m0.val
m0.vct

% Test for unit data
n = 10;
N = 1;  % no observations
obs1 = rand(n,N);
m1 = EigModel_make( obs1 );

fprintf('zero model\n');
m1.N
m1.org
m1.val
m1.vct

% Test for sufficient number of observations data N >= n
n = 10;
N = 100;  % no observations
obsn = rand(n,N);
mn = EigModel_make( obsn );

fprintf('mn model\n');
mn.N
mn.org
mn.val
mn.vct

% Test for in-sufficient number of observations data N < n
n = 10;
N = 5;  % no observations
obsN = rand(n,N);

mN = EigModel_make( obsN );
fprintf('MN model\n');
mN.N
mN.org
mN.val
mN.vct

org = mean(obsN,2);
Y = obsN-repmat(org,1,N);
C = Y*Y'/N;
[U,L] = eig(C);
[L,I] = sort(diag(L));
L = flipud(L);
U = U(:,flipud(I));

% deflated model
mN1 = EigModel_make( obsN, 'keepf', 0.99 );
fprintf('zero model\n');
mN1.N
mN1.org
mN1.val
mN1.vct


