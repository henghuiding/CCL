function [net, softmax_outs,classifier_outs] = Softmax_skipNetwork(net, layer_in, ...
     nh0, nh, nClass, newLr, layer_prefix)
 
 num_skips = numel(layer_in);
 classifier_outs = cell(1, num_skips);
 softmax_outs=cell(1,num_skips);
 for i = 1 : num_skips
    conv_layer = sprintf('%s_conv_%d', layer_prefix, i);
    relu_layer = sprintf('%s_relu_%d',layer_prefix, i);
    drop_layer = sprintf('%s_drop_%d',layer_prefix, i);
    bn_layer = sprintf('%s_bn_%d',layer_prefix, i);
    conv_out = sprintf('%s_conv_out_%d', layer_prefix, i);
    relu_out = sprintf('%s_relu_out_%d', layer_prefix, i);
    drop_out = sprintf('%s_drop_out_%d', layer_prefix, i);
    bn_out = sprintf('%s_bn_out_%d', layer_prefix, i);
    
    conv_param_f = sprintf('%s_cw_f_%d', layer_prefix, i);
    conv_param_b = sprintf('%s_cw_b_%d', layer_prefix, i);
    conv_f = 1e-2*randn(1, 1, nh0, nh, 'single');
    conv_b = zeros(1, 1, nh, 'single');
    
    conv_in = layer_in{i};
    
         %% conv layer
    net.addLayer(conv_layer, ...
        dagnn.Conv('size', [1 1 nh0 nh], 'pad', 0), ...
        conv_in, conv_out, {conv_param_f,conv_param_b});
    
    f = net.getParamIndex(conv_param_f) ;
    net.params(f).value = conv_f ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(conv_param_b) ;
    net.params(f).value = conv_b ;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    %% Batch Normalization
    bn_param_f = sprintf('%s_bn_f_%d', layer_prefix, i);
    bn_param_b = sprintf('%s_bn_b_%d', layer_prefix, i);
    bn_param_m = sprintf('%s_bn_m_%d', layer_prefix, i);
    
    net.addLayer(bn_layer, ...
        dagnn.BatchNorm(), ...
        conv_out, bn_out, {bn_param_f, bn_param_b, bn_param_m});
    
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
    
    %% ReLU
    net.addLayer(relu_layer, ...
        dagnn.ReLU(),...
        bn_out, relu_out);
       
    %% dropout
%     net.addLayer(drop_layer, ...
%         dagnn.DropOut( 'rate', 0.5),...
%         relu_out, drop_out);
    
    %% add an output layer 
    
    classifier = sprintf('%s_classifier_%d', layer_prefix, i);
    classifier_out = sprintf('%s_classifier_out_%d', layer_prefix, i);
    
    classifier_param_f = sprintf('%s_cw_classifier_f_%d', layer_prefix, i);
    classifier_param_b = sprintf('%s_cw_classifier_b_%d', layer_prefix, i);
    classifier_f = zeros(1, 1, nh, nClass, 'single');
    classifier_b = zeros(1, 1, nClass, 'single');
    
    net.addLayer(classifier, ...
        dagnn.Conv('size', [1 1 nh nClass], 'pad', 0), ...
        relu_out, classifier_out, {classifier_param_f,classifier_param_b});
    
    f = net.getParamIndex(classifier_param_f) ;
    net.params(f).value = classifier_f;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(classifier_param_b) ;
    net.params(f).value = classifier_b;
    net.params(f).learningRate = 2  * newLr;
    net.params(f).weightDecay = 1 ;
    
    classifier_outs{i} = classifier_out;
    
    
    softmax = sprintf('%s_softmax_%d', layer_prefix, i);
    softmax_out = sprintf('%s_softmax_out_%d', layer_prefix, i);   
    net.addLayer(softmax, dagnn.SoftMax(), classifier_out, softmax_out);
    softmax_outs{i}=softmax_out;
 end