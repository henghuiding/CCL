function net = fcnInitializeResNetwork(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16

opts.sourceModelPath = '../imagenet/imagenet-resnet-50-dag.mat' ;
opts.rnn = false;
opts.layers = 1;
opts.kerSize = 3;
opts.nh = 512;
opts.nClass = 150;
opts.recursive = false;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;
net = dagnn.DagNN.loadobj(load(opts.sourceModelPath)) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------
% Number of classes
nClass = opts.nClass;
nh = opts.nh;

net.removeLayer('prob');
% net.removeLayer('fc365');
net.removeLayer('fc1000');
net.removeLayer('pool5');
% net.removeLayer('relu5');
% net.removeLayer('bn5');

% adapat the network

net.addLayer('adaptation', ...
     dagnn.Conv('size', [1 1 2048 nh], 'pad', 0), ...
     'res5cx', 'res6x', {'adaptation_f','adaptation_b'});

f = net.getParamIndex('adaptation_f') ;
net.params(f).value = 1e-2*randn(1, 1, 2048, nh, 'single') ;
net.params(f).learningRate = 1 * opts.newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('adaptation_b') ;
net.params(f).value = zeros(1, 1, nh, 'single') ;
net.params(f).learningRate = 2 * opts.newLr ;
net.params(f).weightDecay = 1 ;

net.addLayer('adapation_relu', ...
        dagnn.ReLU(),...
        'res6x', 'res6x1');
layer_in='res6x1';

%% build context network
% [net, layer_out, cn_classifier_out] = contextNetwork_original(net, 'res6x1', opts.kerSize,...
%     nh, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);

[net, layer_out, cn_classifier_out] = CCL(net, 'res6x1', ...
    3, nh, nh, nClass, 6, opts.newLr, 'CCL', opts.recursive,[5 5 5 5 5 5 5], 0);
% [net,layer_out, cn_classifier_out]=ResNet_Block5_nh(net,'res5cx', 6 , 5, 2048, 1024, nh, opts.newLr, nClass);
% [net, ASPPclassifier_out] = ASPP(net, layer_out, 1024, 512, nClass, opts.newLr);
%% build skip network
% skip_inputs = {};
cn_classifier_out=flip(cn_classifier_out);
layer_out=flip(layer_out);
skip_inputs = {'res5cx', 'res5bx', 'res5ax'};
[net, skip_classifier_out] = skipNetwork(net, skip_inputs, 2048, nh, ...
    nClass, opts.newLr, 'skip5');

[net, ~, gate1] = short2_skipNetwork_Sigmoid(net, layer_out, 512, 1, 1, opts.newLr, 'gate1_1', 0);
[net, ~, gate2] = short2_skipNetwork_Sigmoid(net, skip_inputs, 2048, 1, 1, opts.newLr, 'gate1_2', 0);

gate=[gate1 gate2];
kersize=1; LR= 1; wd=0.1; layer_prefix='RNN1'; rnn_in=[]; 
[net, rnn_output] = gated_sum_rnn(net, gate, kersize, LR, wd, layer_prefix, rnn_in, 1);


net.addLayer('rnn_outputs_1',dagnn.Concat(),rnn_output, 'rnn_outputs_1');

layer_in='rnn_outputs_1'; nnh=numel(rnn_output); Lr=0.1; layer_prefix='GlbRfine1';
[net, layer_out] = GlobalRefine(net, layer_in, nnh, Lr, layer_prefix);

% net.addLayer('Artanh1',dagnn.Artanh('slope',0.1),'rnn_outputs_1','Artanh1');
net.addLayer('rnn1_SoftMax',dagnn.SoftMax(), layer_out,'rnn1_SoftMax');
%%
% -------------------------------------------------------------------------
%  Summing layer
% -------------------------------------------------------------------------
net.addLayer('Gated_sum1', ...
    DagGatedsum2('method', 'sum'), ...
    [cn_classifier_out, skip_classifier_out, 'rnn1_SoftMax'], 'sum_1_out','GatedSum1_param');
f = net.getParamIndex('GatedSum1_param') ;
a=ones(1,1,numel([cn_classifier_out, skip_classifier_out])) ;
net.params(f).value = a ;
net.params(f).learningRate = 0.00 ;
net.params(f).weightDecay = 0.000 ;
% net.addLayer('sum_1_1', dagnn.Sum(),  [skip_classifier_out ,cn_classifier_out], 'sum_1_out') ;
deconv_in = 'sum_1_out';


% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------


filters = single(bilinear_u(32, nClass, nClass)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 16, ...
  'crop', 8, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
   deconv_in, 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;


filters = 1.6*single(bilinear_u(4, 1, 1)) ;
net.addLayer('deconvRNN1', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', 1, ...
  'hasBias', false), ...
   rnn_output{end}, 'RNN1_out', 'deconvf_RNN1') ;

f = net.getParamIndex('deconvf_RNN1') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;



% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
  WeightSegmentationLoss('loss', 'idfsoftmaxlog'), ...
  {'prediction', 'label', 'classWeight'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

if 0
  figure(100) ; clf ;
  n = numel(net.vars) ;
  for i=1:n
    vl_tightsubplot(n,i) ;
    showRF(net, 'input', net.vars(i).name) ;
    title(sprintf('%s', net.vars(i).name)) ;
    drawnow ;
  end
end




