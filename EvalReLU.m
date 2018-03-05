function yModel= EvalReLU( indata, Weights)
% Sven Ahlinder Volvo 2018
% yModel=( ReLU([ ones( examples, 1),indata]*  Weights{1}))* Weights{2}
[ examples]= size( indata, 1);
HiddenLayer=  [ ones( examples, 1),indata]*  Weights{1};
HiddenLayer( HiddenLayer< 0)= 0; %ReLU
yModel= [ HiddenLayer]* Weights{2};


