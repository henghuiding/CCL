function [net, layer_out] = feedback_BDR(net, layer_in1, layer_in2, poolsize, layer_prefix)

softmax_layer = sprintf('sftmx_%s', layer_prefix);
softmax_out = sprintf('sftmx_%s', layer_prefix);
prediction_layer = sprintf('pred_%s', layer_prefix);
prediction_out = sprintf('pred_%s', layer_prefix);


BondryDetiction_layer = sprintf('Boundary_%s', layer_prefix);
BondryDetiction_out = sprintf('Boundary_%s', layer_prefix);
multiply_layer = sprintf('filter_%s', layer_prefix);
multiply_out = sprintf('filter_%s', layer_prefix);

net.addLayer(softmax_layer,dagnn.SoftMax(), layer_in1, softmax_out);

net.addLayer(prediction_layer, Prediction(), softmax_out, prediction_out);
minus_outs=cell(1,numel(poolsize));
for i=1:numel(poolsize)
    pooling_layer = sprintf('pooling%d_%s', poolsize(i), layer_prefix);
    pooling_out = sprintf('pooling%d_%s', poolsize(i), layer_prefix);
    minus_layer = sprintf('minus%d_%s', poolsize(i), layer_prefix);
    minus_out = sprintf('minus%d_%s', poolsize(i), layer_prefix);
    net.addLayer(pooling_layer,dagnn.Pooling('poolSize', poolsize(i),'pad', floor(poolsize(i)/2), 'stride',1,...
        'method','avg'), prediction_out, pooling_out);
    
    net.addLayer(minus_layer, dagnn.Minus(), {prediction_out, pooling_out}, minus_out);
    minus_outs{i} = minus_out;
    if numel(poolsize)>1
        net.addLayer('sum4', dagnn.Sum(), minus_outs, 'x12') ;
    end
end

net.addLayer(BondryDetiction_layer, BondryDetiction(), minus_out, BondryDetiction_out);
net.addLayer(multiply_layer, multiply(),{BondryDetiction_out, layer_in2}, multiply_out);
layer_out = multiply_out;


end