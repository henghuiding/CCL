function net = fcnInitializeNetwork8s(net, varargin)
opts.rnn = false;
opts.nh = 256;
opts.nClass = 150;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;
% opts.newLr = 0.01;

%% Remove the last layer
net.removeLayer('deconv16') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x38', 'x39', 'deconvf_2') ;

f = net.getParamIndex('deconvf_2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% build skip network
skip_inputs = {'x23' ,'x21','x19'};
% skip_inputs = {};
[net, skip_classifier_out_1] = skipNetwork(net, skip_inputs, 512, nh, ...
    nClass, opts.newLr, 'skip3_1');

skip_inputs = {'x17'};
[net, skip_classifier_out_2] = skipNetwork(net, skip_inputs, 256, nh, ...
    nClass, opts.newLr, 'skip3_2');

% Add summation layer
net.addLayer('sum3', dagnn.Sum(), ['x39', skip_classifier_out_1, ...
    skip_classifier_out_2], 'x42') ;

% net.addLayer('sum3', DropSum('rate', 0.5), ['x39', skip_classifier_out_1, ...
%     skip_classifier_out_2], 'x42') ;

%% Add deconvolution layers
filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x42', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

%% add conv layers to mimic the crf
% net.addLayer('crf_1', ...
%      dagnn.Conv('size', [3 3 nClass nClass], 'pad', 1), ...
%      'prediction_1', 'prediction', {'crf_f1','crf_b1'});
% 
% f = net.getParamIndex('crf_f1') ;
% net.params(f).value = 1e-2*randn(3, 3, nClass, nClass, 'single');
% net.params(f).learningRate = 1;
% net.params(f).weightDecay = 1 ;
% 
% f = net.getParamIndex('crf_b1') ;
% net.params(f).value = zeros(1, 1, nClass, 'single') ;
% net.params(f).learningRate = 2 ;
% net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;
