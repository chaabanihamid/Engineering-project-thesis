%{
This program aims to train neuron network in order to estimate the snow 
water  equivalent (swe) over land using microwave brightness temperatures
as input.  It also allows you to calculate performance indices and 
visualize model sensitivity analysis figures. 

======= Important note ===========
You should have Matlab installed and activated on your machine.
==================================

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@auteur: hamichaabani@gmail.com (CHAABANI Hamid)
%}

%==== testing a NN with Hamids dataset


%=== Reading data and placing in a matrix

% 'lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'
% Variables explicatives : 'tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'
% Target : SWE (sd)
dirls = dir( [ pwd, '/amsr*'] );
data = [];

for n = 1:length( dirls )
    idata = readmatrix( dirls(n).name );
    data  = [ data; idata ];
    clear idata
end


ind  = find( data(:,4) <= 0.5 );
data = data(ind,:);

%==== random permutation of data to mix the data
ind   = randperm( size(data,1) );
data  = data( ind, : );
	

%=== leaving some data out for my tests on nn performace	 
pdata = data( 1:500e3, : );
data  = data( 500.001e3:end, :);

%=== Sub-setting matrix for NN training
inputs  = data( :, [5:end] )';
targets = data( :, 4 )';



%=== Setting up a second NN of two hidden layer with 14 and 7 nodes
netdou = feedforwardnet( [64 64] );

%= modifying default values
% activation functions hidden layer
netdou.layers{1}.transferFcn = 'tansig';
netdou.layers{2}.transferFcn = 'tansig';

% activation functions output layer
netdou.layers{3}.transferFcn = 'tansig';

% training function
netdou.trainFcn = 'trainlm';
netdou.trainParam.min_grad = 1e-100; 

% maximum epochs
netdou.trainParam.epochs=1000;

% standardizing inputs and outputs before training
netdou.input.processFcns{1}  = 'mapminmax';
netdou.output.processFcns{1} = 'mapminmax';
%netdou.input.processFcns{1}  = 'mapstd';
%netdou.output.processFcns{1} = 'mapstd';
% dividing dataset into the training one (50%, to calibarte the weights), 
% the validation one (25%, to halt the training), and the test one (25%, 
% to get the inversio statistics in an independent dataste not used during
% training or validation)

netdou.divideParam.trainRatio = 0.8;
netdou.divideParam.valRatio   = 0.15;
netdou.divideParam.testRatio  = 0.05;

% error function to drive the weights calibration
netdou.performFcn = 'mse';

% training halted by "early stopping". the MSE
% in the validation is calculated and when it is 
% larger than the MSE in the training data 4 consecutive
% epochs the training is halted
netdou.trainParam.max_fail=5;
netdou = train(netdou,inputs,targets);

%=== Inversion statistics
%= net output on saved pdata
pinput  = pdata( :, [5:end] )';
ptarget = pdata( :,  4 )';
ppre    = netpre( pinput );
pdou    = netdou( pinput );
x{1}    = ptarget;
x{2}    = ptarget; 
xmin = 0;
xmax = 0.5;
xint = 0.05;
xlab = 'SD (m)';
ylab = 'SD_N_N - SD_T_R_U_E (m)';


y{1} = ppre - ptarget;
y{2} = pdou - ptarget;
ymin = -0.5;
ymax = 0.5;

colora  = 'b';
colorb  = 'r';


fig_aux_double_error_bar( x, xmin, xmax, xint, y, ymin, ymax, xlab, ylab, colora, colorb )
legend('1-hidden-layer','2-hiden-layer')
title('Mean [dot] and STD [arrow centered around the dot] at different SD values')

return


performance = perform(netpre,t,y)
% View the Network
view(net)

  outputs = netlin(X(indv,:)');
  errorNN = outputs - LST_tot(indv)';
  me      = mean(errorNN);
  st      = std(errorNN);
  disp([ 'No poles 37 + emis Lin Mean=', num2str(me), '-Std=', num2str(st)]);

