function [net,layer_out ,classifier_outs]=ResNet_Block5_nh(net,layer_in,nlayers, ker_size, nh0, nh1, nh2, newLr, nClass)

layer_prefix=['d','e','f','g','h','i','j','k','l','m'];
conv_in=layer_in;
bn_out0=layer_in;
skip_inputs={};
% nh0 the original input channel num, nh1 the new channel num, nh0 the
% hidden later channel num
if nh1~=nh0
    conv_layer0 = sprintf('res5%s_branch1', layer_prefix(1));
    bn_layer0 = sprintf('bn5%s_branch1', layer_prefix(1));
    conv_out0 = sprintf('res5%s_branch1', layer_prefix(1));
    bn_out0 = sprintf('res5%s_branch1x', layer_prefix(1));
    
    %% conv layer
    cw_param_f = sprintf('res5%s_branch1_filter', layer_prefix(1));
    cw_f_shared = 1e-2*randn(1, 1, 2048, nh1, 'single');
    
    net.addLayer(conv_layer0, ...
        dagnn.Conv('size', [1 1 2048 nh1], 'pad', 0,'hasBias',0), ...
        conv_in, conv_out0, {cw_param_f});
    
    f = net.getParamIndex(cw_param_f) ;
    net.params(f).value = cw_f_shared ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    %% Batch Normalization
    bn_param_f = sprintf('bn5%s_branch1_mult', layer_prefix(1));
    bn_param_b = sprintf('bn5%s_branch1_bias', layer_prefix(1));
    bn_param_m = sprintf('bn5%s_branch1_moments', layer_prefix(1));
    
    net.addLayer(bn_layer0, ...
        dagnn.BatchNorm(), ...
        conv_out0, bn_out0, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh1, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh1, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh1, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
    
    
end
for i=1:nlayers
    conv_layer = sprintf('res5%s_branch2a', layer_prefix(i));
    bn_layer = sprintf('bn5%s_branch2a', layer_prefix(i));
    relu_layer = sprintf('res5%s_branch2a_relu',layer_prefix(i));
    conv_layer2 = sprintf('res5%s_branch2b', layer_prefix(i));
    bn_layer2 = sprintf('bn5%s_branch2b', layer_prefix(i));
    relu_layer2 = sprintf('res5%s_branch2b_relu',layer_prefix(i));
    conv_layer3 = sprintf('res5%s_branch2c', layer_prefix(i));
    bn_layer3 = sprintf('bn5%s_branch2c', layer_prefix(i));
    sum_layer = sprintf('res5%s', layer_prefix(i));
    relu_layer3 = sprintf('res5%s_relu', layer_prefix(i));

    conv_out = sprintf('res5%s_branch2a', layer_prefix(i));
    bn_out = sprintf('res5%s_branch2ax', layer_prefix(i));
    relu_out = sprintf('res5%s_branch2axxx', layer_prefix(i));
    conv_out2 = sprintf('res5%s_branch2b', layer_prefix(i));
    bn_out2 = sprintf('res5%s_branch2bx', layer_prefix(i));
    relu_out2 = sprintf('res5%s_branch2bxxx', layer_prefix(i));
    conv_out3 = sprintf('res5%s_branch2c', layer_prefix(i));
    bn_out3 = sprintf('res5%s_branch2cx', layer_prefix(i));
    sum_out = sprintf('res5%s', layer_prefix(i));
    relu_out3 = sprintf('res5%sx', layer_prefix(i));
    
    
    
   
    %% conv layer
    cw_param_f = sprintf('res5%s_branch2a_filter', layer_prefix(i));
    cw_f_shared = 1e-2*randn(1, 1, nh0, nh2, 'single');
    
    net.addLayer(conv_layer, ...
        dagnn.Conv('size', [1 1 nh0 nh2], 'pad', 0,'hasBias',0), ...
        conv_in, conv_out, {cw_param_f});
    
    f = net.getParamIndex(cw_param_f) ;
    net.params(f).value = cw_f_shared ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
        %% Batch Normalization
    bn_param_f = sprintf('bn5%s_branch2a_mult', layer_prefix(i));
    bn_param_b = sprintf('bn5%s_branch2a_bias', layer_prefix(i));
    bn_param_m = sprintf('bn5%s_branch2a_moments', layer_prefix(i));
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        conv_out, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh2, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh2, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh2, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
        %% ReLU
    net.addLayer(relu_layer,dagnn.ReLU(),bn_out, relu_out);
    
    %% conv layer 2b
    cw_param_f = sprintf('res5%s_branch2b_filter', layer_prefix(i));
    cw_f_shared = 1e-2*randn(ker_size, ker_size, nh2, nh2, 'single');
    
    net.addLayer(conv_layer2, ...
        dagnn.Conv('size', [ker_size ker_size nh2 nh2], 'pad', floor(ker_size/2),'hasBias',0), ...
        relu_out, conv_out2, {cw_param_f});
    
    f = net.getParamIndex(cw_param_f) ;
    net.params(f).value = cw_f_shared ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
        %% Batch Normalization 2b
    bn_param_f = sprintf('bn5%s_branch2b_mult', layer_prefix(i));
    bn_param_b = sprintf('bn5%s_branch2b_bias', layer_prefix(i));
    bn_param_m = sprintf('bn5%s_branch2b_moments', layer_prefix(i));
    
    net.addLayer(bn_layer2, ...
        dagnn.BatchNorm(), ...
        conv_out2, bn_out2, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh2, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh2, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh2, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
        %% ReLU 2b
    net.addLayer(relu_layer2,dagnn.ReLU(),bn_out2, relu_out2);
    
        %% conv layer 2c
    cw_param_f = sprintf('res5%s_branch2c_filter', layer_prefix(i));
    cw_f_shared = 1e-2*randn(1, 1, nh2, nh1, 'single');
    
    net.addLayer(conv_layer3, ...
        dagnn.Conv('size', [1 1 nh2 nh1], 'pad', 0,'hasBias',0), ...
        relu_out2, conv_out3, {cw_param_f});
    
    f = net.getParamIndex(cw_param_f) ;
    net.params(f).value = cw_f_shared ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
        %% Batch Normalization 2c
    bn_param_f = sprintf('bn5%s_branch2c_mult', layer_prefix(i));
    bn_param_b = sprintf('bn5%s_branch2c_bias', layer_prefix(i));
    bn_param_m = sprintf('bn5%s_branch2c_moments', layer_prefix(i));
    
    net.addLayer(bn_layer3, ...
        dagnn.BatchNorm(), ...
        conv_out3, bn_out3, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh1, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh1, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh1, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% sum
    net.addLayer(sum_layer, dagnn.Sum(), {bn_out0, bn_out3}, sum_out) ;
    %% ReLU
    net.addLayer(relu_layer3,dagnn.ReLU(),sum_out, relu_out3);
    conv_in=relu_out3;
    skip_inputs=[skip_inputs,relu_out3];
    nh0=nh1;
    bn_out0=relu_out3;
    
end
layer_out=relu_out3;
    [net, classifier_outs] = skipNetwork(net, skip_inputs, nh1, nh2, ...
    nClass, newLr, 'skip_newblock5');

