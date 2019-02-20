function [net, layer_out, classifier_outs] = PCLA(net, layer_in, ...
    ker_size, nh0, nh, nClass, layers, newLr, layer_prefix, recursive,dilate,residual)
classifier_outs = [];
layer_out={};
if layers == 0
    layer_out = layer_in;
    return;
end

% shared version for recursive net

if recursive
    cw_param_f1 = sprintf('%s_cw_f_shared', layer_prefix);
    cw_param_b1 = sprintf('%s_cw_b_shared', layer_prefix);
    cw_f_shared1 = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
    cw_b_shared1 = zeros(1, 1, nh, 'single');
    
    
    classifier_f = sprintf('%s_cw_classifier_f_shared', layer_prefix);
    classifier_b = sprintf('%s_cw_classifier_b_shared', layer_prefix);
    classifier_f_shared = 1e-2*randn(1, 1, nh, nClass, 'single');
    classifier_b_shared = zeros(1, 1, nClass, 'single');
end


for i = 1 : layers
    if i == 1,
        conv_in = layer_in;
    end
        
    conv_layer1 = sprintf('%s_LC_conv_%d', layer_prefix, i);
    relu_layer1 = sprintf('%s_LC_relu_%d',layer_prefix, i);
    conv_out1 = sprintf('%s_LC_out_%d', layer_prefix, i);
    relu_out1 = sprintf('%s_LC_relu_out_%d', layer_prefix, i);
    bn_layer1 = sprintf('%s_LC_bn_%d', layer_prefix, i);
    bn_out1 = sprintf('%s_LC_bn_out_%d', layer_prefix, i);
    
    if ~recursive
        cw_param_f1 = sprintf('%s_LC_f_%d', layer_prefix, i);
        cw_param_b1 = sprintf('%s_LC_b_%d', layer_prefix, i);
        cw_f_shared1 = 1e-2*randn(ker_size, ker_size, nh0, nh, 'single');
        cw_b_shared1 = zeros(1, 1, nh, 'single');
    end
    
    
    %% conv layer
    net.addLayer(conv_layer1, ...
        dagnn.Conv('size', [ker_size ker_size nh0 nh], 'pad', floor(ker_size/2)), ...
        conv_in, conv_out1, {cw_param_f1,cw_param_b1});
    
    f = net.getParamIndex(cw_param_f1) ;
    net.params(f).value = cw_f_shared1 ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b1) ;
    net.params(f).value = cw_b_shared1 ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    %% Batch Normalization
    bn_param_f1 = sprintf('%s_LC_bn_f_%d', layer_prefix, i);
    bn_param_b1 = sprintf('%s_LC_bn_b_%d', layer_prefix, i);
    bn_param_m1 = sprintf('%s_LC_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer1, ...
        dagnn.BatchNorm(), ...
        conv_out1, bn_out1, {bn_param_f1, bn_param_b1, bn_param_m1});
    
    f = net.getParamIndex(bn_param_f1) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b1) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m1) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% ReLU
    net.addLayer(relu_layer1, ...
        dagnn.ReLU(),...
        bn_out1, relu_out1);
    
    
    conv_layer2 = sprintf('%s_CT_conv_%d', layer_prefix, i);
    relu_layer2 = sprintf('%s_CT_relu_%d',layer_prefix, i);
    conv_out2 = sprintf('%s_CT_out_%d', layer_prefix, i);
    relu_out2 = sprintf('%s_CT_relu_out_%d', layer_prefix, i);
    bn_layer2 = sprintf('%s_CT_bn_%d', layer_prefix, i);
    bn_out2 = sprintf('%s_CT_bn_out_%d', layer_prefix, i);
    
    if ~recursive
        cw_param_f2 = sprintf('%s_CT_f_%d', layer_prefix, i);
        cw_param_b2 = sprintf('%s_CT_b_%d', layer_prefix, i);
        cw_f_shared2 = 1e-2*randn(3, 3, nh0, nh, 'single');
        cw_b_shared2 = zeros(1, 1, nh, 'single');
    end
    
    
    %% conv layer
    net.addLayer(conv_layer2, ...
        dagnn.Conv('size', [3 3 nh0 nh], 'pad', floor((3+(dilate(i)-1)*2)/2),'dilate',dilate(i)), ...
        conv_in, conv_out2, {cw_param_f2,cw_param_b2});
    
    f = net.getParamIndex(cw_param_f2) ;
    net.params(f).value = cw_f_shared2 ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b2) ;
    net.params(f).value = cw_b_shared2 ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    %% Batch Normalization
    bn_param_f2 = sprintf('%s_CT_bn_f_%d', layer_prefix, i);
    bn_param_b2 = sprintf('%s_CT_bn_b_%d', layer_prefix, i);
    bn_param_m2 = sprintf('%s_CT_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer2, ...
        dagnn.BatchNorm(), ...
        conv_out2, bn_out2, {bn_param_f2, bn_param_b2, bn_param_m2});
    
    f = net.getParamIndex(bn_param_f2) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b2) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m2) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% ReLU
    net.addLayer(relu_layer2, ...
        dagnn.ReLU(),...
        bn_out2, relu_out2);
    

    
    sum_layer = sprintf('%s_PCLA_minus_%d', layer_prefix, i);
    sum_out = sprintf('%s_PCLA_minus_out_%d', layer_prefix, i);
    if residual
        net.addLayer(sum_layer, dagnn.Sum(), {relu_out1, relu_out2, conv_in}, sum_out) ;
    else
        net.addLayer(sum_layer, dagnn.Minus(), {relu_out1, relu_out2}, sum_out) ;
    end
    conv_in = sum_out; % input for next conv layer
    nh0 = nh;
    
    %% dropout
%     net.addLayer(drop_layer, ...
%         dagnn.DropOut( 'rate', i/layers *  0.5),...
%         relu_out, drop_out);
%      
    %% add an output layer
    
    [net, classifier_out] = skipNetwork(net, {sum_out}, nh, nh, ...
    nClass, newLr, sprintf('%s_skip5_%d',layer_prefix,i));
    
%     classifier = sprintf('%s_cw_classifier_%d', layer_prefix, i);
%     classifier_out = sprintf('%s_cw_classifier_out_%d', layer_prefix, i);
%     
%     if ~recursive
%         classifier_f = sprintf('%s_cw_classifier_f_%d', layer_prefix, i);
%         classifier_b = sprintf('%s_cw_classifier_b_%d', layer_prefix, i);
%         classifier_f_shared = 1e-2*randn(1, 1, nh, nClass, 'single');
%         classifier_b_shared = zeros(1, 1, nClass, 'single');
%     end
%     
%     
%     net.addLayer(classifier, ...
%         dagnn.Conv('size', [1 1 nh nClass], 'pad', 0), ...
%         relu_out, classifier_out, {classifier_f,classifier_b});
%     
%     f = net.getParamIndex(classifier_f) ;
%     net.params(f).value = classifier_f_shared;
%     net.params(f).learningRate = 1 * newLr ;
%     net.params(f).weightDecay = 1 ;
%     
%     f = net.getParamIndex(classifier_b) ;
%     net.params(f).value = classifier_b_shared;
%     net.params(f).learningRate = 2  * newLr;
%     net.params(f).weightDecay = 1 ;
    
    
%     classifier_out = mat2cell(classifier_out, 1);
    classifier_outs = [classifier_outs, classifier_out];
    layer_out = [layer_out sum_out];
end
