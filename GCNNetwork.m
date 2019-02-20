function [net,layer_out] = GCNNetwork(net, layer_in, ...
    ker_size, nh0, nh, nClass, newLr, layer_prefix)


% shared version for recursive net



    
    bn_layer0 = sprintf('%s_cw_bn_0', layer_prefix);
    bn_out0 = sprintf('%s_bn_out_0', layer_prefix);
    %% 1a
    conv_layer1a = sprintf('%s_cw_conv_1a', layer_prefix);
    relu_layer1a = sprintf('%s_cw_relu_1a',layer_prefix);
    conv_out1a = sprintf('%s_conv_out_1a', layer_prefix);
    relu_out1a = sprintf('%s_relu_out_1a', layer_prefix);
    bn_layer1a = sprintf('%s_cw_bn_1a', layer_prefix);
    bn_out1a = sprintf('%s_bn_out_1a', layer_prefix);
    %% 1b
    conv_layer1b = sprintf('%s_cw_conv_1b', layer_prefix);
    conv_out1b = sprintf('%s_conv_out_1b', layer_prefix);
    bn_layer1b = sprintf('%s_cw_bn_1b', layer_prefix);
    bn_out1b = sprintf('%s_bn_out_1b', layer_prefix);
    %% 2a
    conv_layer2a = sprintf('%s_cw_conv_2a', layer_prefix);
    relu_layer2a = sprintf('%s_cw_relu_2a',layer_prefix);
    conv_out2a = sprintf('%s_conv_out_2a', layer_prefix);
    relu_out2a = sprintf('%s_relu_out_2a', layer_prefix);
    bn_layer2a = sprintf('%s_cw_bn_2a', layer_prefix);
    bn_out2a = sprintf('%s_bn_out_2a', layer_prefix);
    %% 2b
    conv_layer2b = sprintf('%s_cw_conv_2b', layer_prefix);
    conv_out2b = sprintf('%s_conv_out_2b', layer_prefix);
    bn_layer2b = sprintf('%s_cw_bn_2b', layer_prefix);
    bn_out2b = sprintf('%s_bn_out_2b', layer_prefix);
    
    
    cw_param_f1a = sprintf('%s_cw_f_1a', layer_prefix);
    cw_param_b1a = sprintf('%s_cw_b_1a', layer_prefix);    
    cw_param_f1b = sprintf('%s_cw_f_1b', layer_prefix);
    cw_param_b1b = sprintf('%s_cw_b_1b', layer_prefix);
    cw_param_f2a = sprintf('%s_cw_f_2a', layer_prefix);
    cw_param_b2a = sprintf('%s_cw_b_2a', layer_prefix);
    cw_param_f2b = sprintf('%s_cw_f_2b', layer_prefix);
    cw_param_b2b = sprintf('%s_cw_b_2b', layer_prefix);
    
    cw_f_shared_a = 1e-2*randn(ker_size, 1, nh0, nh, 'single');
    cw_b_shared_a = zeros(1, 1, nh, 'single');
    cw_f_shared_b = 1e-2*randn(1, ker_size, nh0, nh, 'single');
    cw_b_shared_b = zeros(1, 1, nh, 'single');

    %% BN_0
    bn_param_f = sprintf('%s_bn_f_0', layer_prefix);
    bn_param_b = sprintf('%s_bn_b_0', layer_prefix);
    bn_param_m = sprintf('%s_bn_m_0', layer_prefix);
    
    net.addLayer(bn_layer0, ...
        dagnn.BatchNorm(), ...
        layer_in, bn_out0, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh0, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh0, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh0, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
    %% conv layer 1a
    net.addLayer(conv_layer1a, ...
        dagnn.Conv('size', [ker_size 1 nh0 nh], 'pad', [floor(ker_size/2),0]), ...
        bn_out0, conv_out1a, {cw_param_f1a,cw_param_b1a});
    
    f = net.getParamIndex(cw_param_f1a) ;
    net.params(f).value = cw_f_shared_a ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b1a) ;
    net.params(f).value = cw_b_shared_a ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    
    %% Batch Normalization 1a
    bn_param_f = sprintf('%s_bn_f_1a', layer_prefix);
    bn_param_b = sprintf('%s_bn_b_1a', layer_prefix);
    bn_param_m = sprintf('%s_bn_m_1a', layer_prefix);
    
    net.addLayer(bn_layer1a, ...
        dagnn.BatchNorm(), ...
        conv_out1a, bn_out1a, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% ReLU 1a
    net.addLayer(relu_layer1a, ...
        dagnn.ReLU(),...
        bn_out1a, relu_out1a);
    
    %% conv layer 1b
    net.addLayer(conv_layer1b, ...
        dagnn.Conv('size', [1 ker_size nh nh], 'pad', [0,floor(ker_size/2)]), ...
        relu_out1a, conv_out1b, {cw_param_f1b,cw_param_b1b});
    
    f = net.getParamIndex(cw_param_f1b) ;
    net.params(f).value = cw_f_shared_b ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b1b) ;
    net.params(f).value = cw_b_shared_b ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    
    %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_1b', layer_prefix);
    bn_param_b = sprintf('%s_bn_b_1b', layer_prefix);
    bn_param_m = sprintf('%s_bn_m_1b', layer_prefix);
    
    net.addLayer(bn_layer1b, ...
        dagnn.BatchNorm(), ...
        conv_out1b, bn_out1b, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
    
    
     %% conv layer 2a
    net.addLayer(conv_layer2a, ...
        dagnn.Conv('size', [1 ker_size nh0 nh], 'pad', [0,floor(ker_size/2)]), ...
        bn_out0, conv_out2a, {cw_param_f2a,cw_param_b2a});
    
    f = net.getParamIndex(cw_param_f2a) ;
    net.params(f).value = cw_f_shared_b ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b2a) ;
    net.params(f).value = cw_b_shared_b ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    
    %% Batch Normalization 2a
    bn_param_f = sprintf('%s_bn_f_2a', layer_prefix);
    bn_param_b = sprintf('%s_bn_b_2a', layer_prefix);
    bn_param_m = sprintf('%s_bn_m_2a', layer_prefix);
    
    net.addLayer(bn_layer2a, ...
        dagnn.BatchNorm(), ...
        conv_out2a, bn_out2a, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    %% ReLU 2a
    net.addLayer(relu_layer2a, ...
        dagnn.ReLU(),...
        bn_out2a, relu_out2a);
    
    %% conv layer 2b
    net.addLayer(conv_layer2b, ...
        dagnn.Conv('size', [ker_size 1 nh nh], 'pad', [floor(ker_size/2),0]), ...
        relu_out2a, conv_out2b, {cw_param_f2b,cw_param_b2b});
    
    f = net.getParamIndex(cw_param_f2b) ;
    net.params(f).value = cw_f_shared_a ;
    net.params(f).learningRate = 1 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(cw_param_b2b) ;
    net.params(f).value = cw_b_shared_a ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    
    %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_2b', layer_prefix);
    bn_param_b = sprintf('%s_bn_b_2b', layer_prefix);
    bn_param_m = sprintf('%s_bn_m_2b', layer_prefix);
    
    net.addLayer(bn_layer2b, ...
        dagnn.BatchNorm(), ...
        conv_out2b, bn_out2b, {bn_param_f, bn_param_b, bn_param_m});
    
    f = net.getParamIndex(bn_param_f) ;
    net.params(f).value = ones(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_b) ;
    net.params(f).value = zeros(nh, 1, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(bn_param_m) ;
    net.params(f).value = zeros(nh, 2, 'single') ;
    net.params(f).learningRate = 0  ;
    net.params(f).weightDecay = 0 ;
    
    sumlayer=sprintf('%s_sum', layer_prefix);
    sumout= sprintf('%s_sum_out', layer_prefix);
    net.addLayer(sumlayer, dagnn.Sum(), {bn_out1b,bn_out2b},...
        sumout) ;
    layer_out=sumout;
    %% add an output layer
    
%     [net, classifier_out] = skipNetwork(net, {sumout}, ...
%      nh, nh, nClass, newLr, sprintf('%s_classifier',layer_prefix));
    
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
%     net.params(f).learningRate = 1  * newLr;
%     net.params(f).weightDecay = 1 ;
%     
%     f = net.getParamIndex(classifier_b) ;
%     net.params(f).value = classifier_b_shared;
%     net.params(f).learningRate = 2  * newLr;
%     net.params(f).weightDecay = 1 ;
%     
%     
%     classifier_out = mat2cell(classifier_out, 1);
%     classifier_outs = [ classifier_outs, classifier_out];


