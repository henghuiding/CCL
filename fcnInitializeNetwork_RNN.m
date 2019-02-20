function net = fcnInitializeNetwork_RNN(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16
% opts.sourceModelPath= '../FCNs-S3/pose/lsp-mirror-crop/net-epoch-11.mat' ;

opts.sourceModelPath = '../imagenet/imagenet-vgg-verydeep-16.mat' ;
opts.rnn = false;
opts.recursive = false;
opts.layers = 1;
opts.kerSize = 3;
opts.nh = 512;
opts.nClass = 150;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;
net = load(opts.sourceModelPath) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------
% Number of classes
% nh = 512;
% nClass = 150;

nh = opts.nh;
nClass = opts.nClass;

%% For imagenet pretrained model
conv51f=net.layers{1,25}.weights{1};conv52f=net.layers{1,27}.weights{1};conv53f=net.layers{1,29}.weights{1};
conv51b=net.layers{1,25}.weights{2};conv52b=net.layers{1,27}.weights{2};conv53b=net.layers{1,29}.weights{2};
net.layers = net.layers(1:23);

% FCN only
% net.layers{32}.pad = [3, 3, 3, 3];

 %Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
% net.removeLayer('pool4');
% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end
%conv5_1
net.addLayer('conv5_1', ...
    dagnn.Conv('size', [3 3 512 512], 'pad', 2,'dilate',2 ),...
    'x23', 'x24', {'conv5_1f','conv5_1b'});
f = net.getParamIndex('conv5_1f') ;
net.params(f).value = conv51f ;
f = net.getParamIndex('conv5_1b') ;
net.params(f).value = conv51b ;

net.addLayer('relu5_1', dagnn.ReLU(),'x24', 'x25');

net.addLayer('conv5_2', ...
    dagnn.Conv('size', [3 3 512 512], 'pad', 2,'dilate',2 ), ...
    'x25', 'x26', {'conv5_2f','conv5_2b'});
f = net.getParamIndex('conv5_2f') ;
net.params(f).value = conv52f ;
f = net.getParamIndex('conv5_2b') ;
net.params(f).value = conv52b ;

net.addLayer('relu5_2', dagnn.ReLU(),'x26', 'x27');

net.addLayer('conv5_3', ...
    dagnn.Conv('size', [3 3 512 512], 'pad', 2,'dilate',2 ), ...
    'x27', 'x28', {'conv5_3f','conv5_3b'});
f = net.getParamIndex('conv5_3f') ;
net.params(f).value = conv53f ;
f = net.getParamIndex('conv5_3b') ;
net.params(f).value = conv53b ;

net.addLayer('relu5_3', dagnn.ReLU(),'x28', 'x29');


%% build context network
[net, layer_out, classifier_out] = contextNetwork_2(net, 'x29', opts.kerSize,...
    512, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive,2*[1 2 1 2 1 2]);

%% build context network
% [net, ~, classifier_out] = contextNetwork(net, 'x31', opts.kerSize,...
%     512, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);

%% build skip network
% skip_inputs = {'x31'};
% [net, skip_classifier_out] = skipNetwork(net, skip_inputs, 512, nh, ...
%     nClass, opts.newLr, 'skip5');
%%
layer_out=flip(layer_out);
skip_inputs = [layer_out, {'x29', 'x27', 'x25', 'x23','x21'}];

[net, skip_classifier_out_1] = skipNetwork(net, skip_inputs, 512, nh, ...
    nClass, opts.newLr, 'skip');
% [net, skip_classifier_out_2] = skipNetwork(net, {'x17'}, 256, nh, ...
%     nClass, opts.newLr, 'skip_2');
% 
% nnh=64;
% [net, gate1] = short2_skipNetwork(net, skip_inputs, 512, nnh, ...
%     1, opts.newLr, 'gate_1');
% [net, gate2] = short2_skipNetwork(net, {'x17'}, 256, nnh, ...
%     1, opts.newLr, 'gate_2');
% net.addLayer('Gate_concat',dagnn.Concat(),[gate1,gate2],'Gate_Concat');
% hh=numel([gate1,gate2])*nnh;
% [net, Gate] = short3_skipNetwork(net, {'Gate_Concat'}, hh, 512 ,numel([gate1,gate2]), opts.newLr, 'Gate_relation');
[net, gate] = short2_skipNetwork_Sigmoid(net, skip_inputs, 512, 1, 1, opts.newLr, 'gate');
conv_f=1e-2*randn(3,3,2,1,'single');
conv_f=0.056+conv_f;
conv_b=zeros(1, 1, 1, 'single');
rnn_output{1}=gate{1};
for ii=2:numel(skip_inputs)
    concat_layer=sprintf('RNN_Concat%d',ii);
    concat_output=sprintf('RNN_Concat%d',ii);
    conv_layer=sprintf('RNN_conv%d',ii);
    conv_out=sprintf('RNN_conv%d',ii);
    conv_param_f=sprintf('RNN_conv%d_f',ii);
    conv_param_b=sprintf('RNN_conv%d_b',ii);
    sigmoid_layer=sprintf('RNN_sigmoid%d',ii);
    sigmoid_out=sprintf('RNN_sigmoid%d',ii);
    net.addLayer(concat_layer,dagnn.Concat(),{rnn_output{ii-1},gate{ii}}, concat_output);
             %% conv layer
    net.addLayer(conv_layer, ...
        dagnn.Conv('size', [3 3 2 1], 'pad', 1), ...
        concat_output, conv_out, {conv_param_f,conv_param_b});
    
    f = net.getParamIndex(conv_param_f) ;
    net.params(f).value = conv_f ;
    net.params(f).learningRate = 1;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(conv_param_b) ;
    net.params(f).value = conv_b ;
    net.params(f).learningRate = 2 ;
    net.params(f).weightDecay = 1 ;
    
    net.addLayer(sigmoid_layer,dagnn.Sigmoid(),conv_out,sigmoid_out);
    rnn_output{ii}=sigmoid_out;
end
net.addLayer('rnn_outputs',dagnn.Concat(),rnn_output, 'rnn_outputs');
% num=numel([classifier_out,skip_inputs,{'x17'}]);
% [net, Gate] = skipNetwork(net, {layer_out}, 512, nh, ...
%     num, opts.newLr, 'gate');
% net.addLayer('Gate_SoftMax',dagnn.SoftMax(),Gate,'Gate_SoftMax');
% -------------------------------------------------------------------------
%  Summing layer
% -------------------------------------------------------------------------
if numel(skip_classifier_out_1) > 0
%     net.addLayer('sum_1_1', dagnn.Sum(), [classifier_out,skip_classifier_out_1,skip_classifier_out_2],...
%         'sum_1_out') ;
net.addLayer('Gated_sum', ...
        DagGatedsum_2('method', 'sum'), ...
        [skip_classifier_out_1, 'rnn_outputs'], 'sum_1_out');
    
%     net.addLayer('sum_1_1', DropSum('rate', 0.5), classifier_out,...
%         'sum_1_out') ;
    
    deconv_in = 'sum_1_out';
else
    error('The depth of context network must be deeper than 1.');
end


%%
% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------

filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  deconv_in, 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

%%
% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
  WeightSegmentationLoss('loss', 'idfsoftmaxlog'), ...
  {'prediction', 'label', 'classWeight'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy('nClass', nClass), ...
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
