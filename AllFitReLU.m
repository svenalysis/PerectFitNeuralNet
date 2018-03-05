function [Weights, HiddenLayer]= AllFitReLU( x, y)
% [RMS, Weights]= AllFitReLU( x, y)
% Sven Ahlinder, Volvo Corporation 2018
%
% A neural network that perfectly fits
% every in-data x to every out-data y
% if there are no repetition error
%
% "x" is in-data with examples in columns and variables in rows
% "y" is out-data with examples in columns and variables in rows
% "Weight" is the values of the weights in the two layers
%
% The idea is to make a single "HiddenLayer" where:
% * number of nodes = "examples of x" * "variables of x"
% * all columns in the matrix "HiddenLayer" are indepenendent
% Then the system of equation Weights{2}= pinv( HiddenLayer)*y; row 44
% always gives a perfect fit

[Weights, HiddenLayer]= FitReLU (x, y);
yModel= EvalReLU( x, Weights);

% RMS= sqrt( sum( (y- yModel).^ 2)/ size( x, 1));
figure;
hold;
for i= 1: size( y, 2)
    plot(y( :, i), yModel( :, i),'*')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Weights, HiddenLayer]= FitReLU (x, y)
[examples, variables]= size( x);

% make as many independent nodes as examples in x
Weights{1}=[];
for i= 1: variables
    
    bias= unique( x( :, i));
    bias= [ -bias(1)+ 1, -bias( 1: end- 1)'];
    w=[ bias; zeros( variables, size( bias, 2))];
    w( i+ 1, :)= 1;
    Weights{1}= [Weights{1} , w];
end
HiddenLayer= [ ones( examples, 1), x]* Weights{1};
HiddenLayer( HiddenLayer< 0)= 0; %ReLU

% training of second weights
Weights{2}= pinv( HiddenLayer)*y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yModel= EvalReLU( indata, Weights)
examples= size( indata, 1);
HiddenLayer= [ ones( examples, 1), indata]* Weights{1};
HiddenLayer( HiddenLayer< 0)= 0; %ReLU
yModel=  HiddenLayer* Weights{2};


