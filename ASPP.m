function [net, classifier_out] = ASPP(net, layer_in, nh0, nh, nClass, newLr)

classifier_out = cell(1,4);
dilations = [1,3,4,6];

for i = 1 : 4
    dilation = dilations(i);
    
    fc6_layer = sprintf('fc6_%d', i);
    relu6_layer = sprintf('relu6_%d', i);
    fc6_out = sprintf('fc6_%d', i);
    relu6_out = sprintf('relu6_%d', i);
    
    fc7_layer = sprintf('fc7_%d', i);
    relu7_layer = sprintf('relu7_%d', i);
    fc7_out = sprintf('fc7_%d', i);
    relu7_out = sprintf('relu7_%d', i);
    
    fc8_layer = sprintf('fc8_%d', i);
    fc8_out = sprintf('fc8_%d', i);
    
    fc6_f = sprintf('fc6_f%d', i);
    fc6_b = sprintf('fc6_b%d', i);
    
    fc7_f = sprintf('fc7_f%d', i);
    fc7_b = sprintf('fc7_b%d', i);
    
    fc8_f = sprintf('fc8_f%d', i);
    fc8_b = sprintf('fc8_b%d', i);
    
     %% fc6
    net.addLayer(fc6_layer, ...
        dagnn.Conv('size', [3 3 nh0 nh], 'pad', dilation, 'dilate', dilation), ...
        layer_in, fc6_out, {fc6_f,fc6_b});
    
    f = net.getParamIndex(fc6_f) ;
    net.params(f).value = 1e-2 * randn(3, 3, nh0, nh, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(fc6_b) ;
    net.params(f).value = zeros(1, 1, nh, 'single') ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
     % ReLU
    net.addLayer(relu6_layer, ...
        dagnn.ReLU(),...
        fc6_out, relu6_out);
    
    %% fc7
    net.addLayer(fc7_layer, ...
        dagnn.Conv('size', [1 1 nh nh]), ...
        relu6_out, fc7_out, {fc7_f,fc7_b});
    
    f = net.getParamIndex(fc7_f) ;
    net.params(f).value = 1e-2 * randn(1, 1, nh, nh, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(fc7_b) ;
    net.params(f).value = zeros(1, 1, nh, 'single') ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
     % ReLU
    net.addLayer(relu7_layer, ...
        dagnn.ReLU(),...
        fc7_out, relu7_out);
    
   %% fc8
    net.addLayer(fc8_layer, ...
        dagnn.Conv('size', [1 1 nh nClass]), ...
        relu7_out, fc8_out, {fc8_f,fc8_b});
    
    f = net.getParamIndex(fc8_f) ;
    net.params(f).value = 1e-2 * randn(1, 1, nh, nClass, 'single') ;
    net.params(f).learningRate = 1  * newLr;
    net.params(f).weightDecay = 1 ;
    
    f = net.getParamIndex(fc8_b) ;
    net.params(f).value = zeros(1, 1, nClass, 'single') ;
    net.params(f).learningRate = 2 * newLr ;
    net.params(f).weightDecay = 1 ;
    
    classifier_out{i} = fc8_out;
end