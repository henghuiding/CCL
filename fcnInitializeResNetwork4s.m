function net = fcnInitializeResNetwork4s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;

%% Remove the last layer
net.removeLayer('deconv8') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'sum_3_out', 'x11', 'deconvf_3') ;

f = net.getParamIndex('deconvf_3') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;


net.addLayer('sftmx_bdr1',dagnn.SoftMax(), 'x11','sftmx_bdr1');

net.addLayer('pred_bdr1', Prediction(),'sftmx_bdr1','pred_bdr1');
%% poolsize3
net.addLayer('pooling5_dbr1',dagnn.Pooling('poolSize', 3,'pad', 1, 'stride',1,...
        'method','avg'),'pred_bdr1', 'pooling5_bdr1');
net.addLayer('minus_bdr1',dagnn.Minus(), {'pred_bdr1', 'pooling5_bdr1'}, 'minus_bdr1');
net.addLayer('bdr1', BondryDetiction('size', 25,'sigma',5,'value', 2), 'minus_bdr1','bdr1');
%% build skip network

skip_inputs = {'res2cx', 'res2bx', 'res2ax'};
        
[net, classifier_out] = skipNetwork(net, skip_inputs, 256, 256, ...
    nClass, opts.newLr, 'skip2');
net.addLayer('sum4', dagnn.Sum(), classifier_out, 'x12') ;
% net.addLayer('sum4', dagnn.Sum(), ['x11',classifier_out], 'sum_4_out') ;

net.addLayer('bdr1_filter',multiply(),{'bdr1','x12'}, 'bdr1_filter');
net.addLayer('sum5', dagnn.Sum(), {'bdr1_filter', 'x11'}, 'sum_4_out') ;


%% Add deconvolution layers
filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv4', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'sum_4_out', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;
