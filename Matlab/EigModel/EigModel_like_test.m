% quickly test for Mahalanobis, likelihood
close all;
clear all;

% % one-point data
% X = rand(2,1);
% emodel = EigModel_make( X );
% Y = rand( 2,1);
% h0 = EigModel_Mahalanobis( [X,Y], emodel );

% make some data
n = 3;        % dimension of each point
N = 1000; % number of points
a = 30*pi/180;
R = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
S = [ 1 0 0; 0 1 0; 0 0 0.01];
org = [pi; -1; 4];
X = R*S*gsamp( zeros(1,n), eye( n), N )' + repmat( org, 1, N );

% plot3( X(1,:), X(2,:), X(3,:), 'k.' );

% make an EigenModel
emodel1 = EigModel_make( X );

% make a deflated version
emodel2 = EigModel_deflate( emodel1, 'keepN', 2 );

% test Mahalanobis distance
h1 = EigModel_Mahalanobis( X, emodel1 );
h2 = EigModel_Mahalanobis( X, emodel2 );
h1a = EigModel_Mahalanobis( X, emodel1, 'MogPen' );
h2a = EigModel_Mahalanobis( X, emodel2, 'MogPen'  );
% [h1(1:5); h2(1:5); h1a(1:5); h2a(1:5)]


[max(  abs(h1 - h2) ) max(  abs(h1 - h2a) )]

% test likelihoods
l1 = EigModel_likelihood( X, emodel1 );
l2 = EigModel_likelihood( X, emodel2 );
l1a = EigModel_likelihood( X, emodel1, 'MogPen' );
l2a = EigModel_likelihood( X, emodel2, 'MogPen'  );

[max(  abs(l1 - l2) ) max(  abs(l1 - l2a) )]



