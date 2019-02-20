
function net = fcnInitializeResNetwork8s_3(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;

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
  'x6', 'x7', 'deconvf_2') ;

f = net.getParamIndex('deconvf_2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

net.addLayer('deconv16_softmax',dagnn.SoftMax(), 'x7', 'x7_softmax');

%% Add direct output from pool4 

switch opts.resLayer
    case 50
        skip3_input = 'res3dx'
    case 101
        skip3_input = 'res3b3x'
    case 152
        skip3_input = 'res3b7x'
end

%% build skip network
skip_inputs = {};
switch opts.resLayer
    case 50
        % 50 layer
        skip_inputs = {'res3bx', 'res3cx'};
        
    case 101
        % 101 layer
        for ll = 1 : 2
                skip_inputs{end+1} = sprintf('res3b%dx',ll);
        end
    case 152
        % 152 layer
        for ll = 1 : 6
                skip_inputs{end+1} = sprintf('res3b%dx',ll);
        end
end

skip_inputs = ['res3ax', skip_inputs,  skip3_input];

[net, softmax_outs, classifier_out] = Softmax_skipNetwork(net, skip_inputs, 512, 512, ...
    nClass, opts.newLr, 'skip3');

% Add summation layer
% net.addLayer('sum3', dagnn.Sum(), ['x7', classifier_out], 'x10') ;

net.addLayer('confidence8',Confidence(), ['x7_softmax',softmax_outs], 'confidence8');

net.addLayer('Gated_sum_3', ...
    DagGatedsum2('method', 'sum'), ...
    ['x7', classifier_out,'confidence8'], 'x10', 'WeightSum_param3');
f = net.getParamIndex('WeightSum_param3') ;
net.params(f).value = rnn_initialize(nClass,numel(['x7', classifier_out]),1) ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 0.001 ;

%% Add deconvolution layers
filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x10', 'prediction_8', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;


% net.addLayer('LabelConflict', LabelConflict(), 'prediction_8', 'prediction','Label_w') ;
% f = net.getParamIndex('Label_w') ;
% net.params(f).value = 1e-3*randn(nClass, nClass, 'single');
% net.params(f).value(logical(eye(size(net.params(f).value)))) = 1;
% net.params(f).learningRate = 10 ;
% net.params(f).weightDecay = 0.01 ;
ker_size=1;
net.addLayer('crf_2', ...
     dagnn.Conv('size', [ker_size ker_size nClass nClass], 'pad', floor(ker_size/2),'hasBias',0), ...
     'prediction_8', 'prediction', {'crf_f2'});

f = net.getParamIndex('crf_f2') ;
net.params(f).value = 1e-5*randn(ker_size, ker_size, nClass, nClass, 'single');
for i=1:nClass
    net.params(f).value(:,:,i,i)=ones(ker_size, ker_size, 1, 1, 'single');
%     net.params(f).value(2,2,i,i)=1;
end
net.params(f).learningRate = 10;
net.params(f).weightDecay = 0.1 ;

% f = net.getParamIndex('crf_b2') ;
% net.params(f).value = zeros(1, 1, nClass, 'single') ;
% net.params(f).learningRate = 6 ;
% net.params(f).weightDecay = 0.1 ;
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
