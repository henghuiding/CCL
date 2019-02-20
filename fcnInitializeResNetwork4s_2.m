function net = fcnInitializeResNetwork4s_2(net, varargin)
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
net.addLayer('bdr1', BondryDetiction('size',25,'sigma',5,'value',2), 'minus_bdr1','bdr1');


skip_inputs = {'res2cx', 'res2bx', 'res2ax'};
        
[net, classifier_out] = skipNetwork(net, skip_inputs, 256, 256, ...
    nClass, opts.newLr, 'skip2');

[net, ~, gate] = short2_skipNetwork_Sigmoid(net, skip_inputs, 256, 1, 1, opts.newLr, 'gate4',0);

kersize=1; LR= 1; wd=0.1; layer_prefix='RNN4'; rnn_in= 'RNN3_out'; 
[net, rnn_output] = gated_sum_rnn(net, gate, kersize, LR, wd, layer_prefix, rnn_in, 1);

net.addLayer('rnn_outputs_4_1', dagnn.Concat(), rnn_output, 'rnn_outputs_4_1');
net.addLayer('BDR_filter',multiply(),{'rnn_outputs_4_1','bdr1'},'BDR_filter');

net.addLayer('rnn_outputs_4',dagnn.Concat(),{'RNN3_out','BDR_filter'}, 'rnn_outputs_4');
% net.addLayer('Artanh3',dagnn.Artanh('slope',0.1),'rnn_outputs_3','Artanh3');
% net.addLayer('BdRefine3',multiply(),{'Artanh3','mask1'},'BdRefine3');

layer_in='rnn_outputs_4'; nnh=numel(['RNN3_out',rnn_output]); Lr=0.1; layer_prefix='GlbRfine4';
[net, layer_out] = GlobalRefine(net, layer_in, nnh, Lr, layer_prefix);

net.addLayer('rnn4_SoftMax',dagnn.SoftMax(),layer_out,'rnn4_SoftMax');
net.addLayer('Gated_sum4', ...
    DagGatedsum2('method', 'sum'), ...
    ['x11', classifier_out, 'rnn4_SoftMax'], 'sum_4_out','GatedSum4_param');
f = net.getParamIndex('GatedSum4_param') ;
a=ones(1,1,numel(['x11', classifier_out])) ;
a(1,1,1)=1;
net.params(f).value = a ;
net.params(f).learningRate = 0.00 ;
net.params(f).weightDecay = 0.000 ;

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
