function [net, layer_out] = GlobalRefine(net, layer_in, nh, newLr, layer_prefix)
% kersize=1;
% LR=0.1; learningRate
% wd=0.1; weightDecay




conv_layer = sprintf('%s_conv', layer_prefix);
sum_layer = sprintf('%s_sum',layer_prefix);
bn_layer = sprintf('%s_bn',layer_prefix);
conv_out = sprintf('%s_conv_out', layer_prefix);
sum_out = sprintf('%s_sum_out', layer_prefix);
bn_out = sprintf('%s_bn_out', layer_prefix);

conv_param_f = sprintf('%s_f', layer_prefix);
conv_param_b = sprintf('%s_b', layer_prefix);
conv_f = zeros(1, 1, nh, nh, 'single');
conv_b = zeros(1, 1, nh, 'single');



%% Batch Normalization
bn_param_f = sprintf('%s_bn_f', layer_prefix);
bn_param_b = sprintf('%s_bn_b', layer_prefix);
bn_param_m = sprintf('%s_bn_m', layer_prefix);

net.addLayer(bn_layer, ...
    dagnn.BatchNorm(), ...
    layer_in, bn_out, {bn_param_f, bn_param_b, bn_param_m});

f = net.getParamIndex(bn_param_f) ;
net.params(f).value = ones(nh, 1, 'single') ;
net.params(f).learningRate = 1 * newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex(bn_param_b) ;
net.params(f).value = zeros(nh, 1, 'single') ;
net.params(f).learningRate = 1  * newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex(bn_param_m) ;
net.params(f).value = zeros(nh, 2, 'single') ;
net.params(f).learningRate = 0  ;
net.params(f).weightDecay = 0 ;

%% conv layer
net.addLayer(conv_layer, ...
    dagnn.Conv('size', [1 1 nh nh], 'pad', 0), ...
    bn_out, conv_out, {conv_param_f,conv_param_b});

f = net.getParamIndex(conv_param_f) ;
net.params(f).value = conv_f ;
net.params(f).learningRate = 1  * newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex(conv_param_b) ;
net.params(f).value = conv_b ;
net.params(f).learningRate = 2  * newLr;
net.params(f).weightDecay = 1 ;

net.addLayer(sum_layer,dagnn.Sum(), {layer_in, conv_out}, sum_out);


layer_out=sum_out;

end