function net = fcnInitializeResNetwork8s_backup(net, varargin)
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
  'sum_2_out', 'x7', 'deconvf_2') ;

f = net.getParamIndex('deconvf_2') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;



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
skip_inputs=flip(skip_inputs);
[net, classifier_out] = skipNetwork(net, skip_inputs, 512, 512, ...
    nClass, opts.newLr, 'skip3');


[net, ~, gate] = short2_skipNetwork_Sigmoid(net, skip_inputs, 512, 1, 1, opts.newLr, 'gate3',0);



%% BDR
net.addLayer('sftmx_bdr8',dagnn.SoftMax(), 'x7','sftmx_bdr8');
net.addLayer('pred_bdr8', Prediction(),'sftmx_bdr8','pred_bdr8');
net.addLayer('pooling7_dbr8',dagnn.Pooling('poolSize', 7,'pad', 3, 'stride',1,...
        'method','avg'),'pred_bdr8', 'pooling5_bdr8');
net.addLayer('minus_bdr8',dagnn.Minus(), {'pred_bdr8', 'pooling5_bdr8'}, 'minus_bdr8');
net.addLayer('bdr8', BondryDetiction2(), 'minus_bdr8','bdr8');
% net.addLayer('bdr8_filter',multiply(),{'bdr8','x12'}, 'bdr8_filter');
%%

kersize=1; LR= 1; wd=0.1; layer_prefix='RNN3'; rnn_in= 'RNN2_out'; 
[net, rnn_output] = gated_sum_rnn(net, gate, kersize, LR, wd, layer_prefix, rnn_in, 1);

net.addLayer('rnn_outputs3',dagnn.Concat(), rnn_output, 'rnn_outputs3');
net.addLayer('bdr8_filter',multiply(),{'bdr8','rnn_outputs3'}, 'bdr8_filter');
net.addLayer('rnn_outputs_3',dagnn.Concat(),{'RNN2_out','bdr8_filter'}, 'rnn_outputs_3');
% net.addLayer('Artanh3',dagnn.Artanh('slope',0.1),'rnn_outputs_3','Artanh3');
% net.addLayer('BdRefine3',multiply(),{'Artanh3','mask1'},'BdRefine3');

% layer_in='rnn_outputs_3'; nnh=numel(['RNN2_out',rnn_output]); Lr=0.1; layer_prefix='GlbRfine3';
% [net, layer_out] = GlobalRefine(net, layer_in, nnh, Lr, layer_prefix);

net.addLayer('rnn3_SoftMax',dagnn.SoftMax(),'rnn_outputs_3','rnn3_SoftMax');
net.addLayer('Gated_sum3', ...
    DagGatedsum2('method', 'sum'), ...
    ['x7', classifier_out, 'rnn3_SoftMax'], 'sum_3_out','GatedSum3_param');
f = net.getParamIndex('GatedSum3_param') ;
a=ones(1,1,numel(['x7', classifier_out])) ;
a(1,1,1)=1;
net.params(f).value = a ;
net.params(f).learningRate = 0.00 ;
net.params(f).weightDecay = 0.000 ;

%% Add deconvolution layers
filters = single(bilinear_u(8, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 4, ...
  'crop', 2, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'sum_3_out', 'prediction', 'deconvf') ;

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
