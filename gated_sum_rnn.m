function [net, rnn_output] = gated_sum_rnn(net, gate, kersize, LR, wd, layer_prefix, rnn_in, shared)

% kersize=1;
% LR=0.1; learningRate
% wd=0.1; weightDecay

conv_f=1e-2*randn(kersize,kersize,2,1,'single');
conv_f(:,:,1)=-0.12;
conv_f(:,:,2)=0.24;
conv_b=zeros(1, 1, 1, 'single');

for ii=1:numel(gate)
    concat_layer=sprintf('%s_Concat%d', layer_prefix, ii);
    concat_output=sprintf('%s_Concat%d', layer_prefix, ii);
    conv_layer=sprintf('%s_conv%d', layer_prefix, ii);
    conv_out=sprintf('%s_conv%d', layer_prefix, ii);
    if shared
        conv_param_f='RNN_shared_f';
        conv_param_b='RNN_shared_b';
    else
        conv_param_f=sprintf('%s_conv%d_f', layer_prefix, ii);
        conv_param_b=sprintf('%s_conv%d_b', layer_prefix, ii);
    end
    tanh_layer=sprintf('%s_tanh%d', layer_prefix, ii);
    tanh_out=sprintf('%s_tanh%d', layer_prefix, ii);
    if ii==1
        if isempty(rnn_in)
            conv_param_f=sprintf('%s_conv%d_f', layer_prefix, ii);
            conv_param_b=sprintf('%s_conv%d_b', layer_prefix, ii);
            %% conv layer
            net.addLayer(conv_layer, ...
                dagnn.Conv('size', [kersize kersize 1 1], 'pad', floor(kersize/2)), ...
                gate{ii}, conv_out, {conv_param_f,conv_param_b});
            
            f = net.getParamIndex(conv_param_f) ;
            net.params(f).value = 0.24+1e-2*randn(kersize, kersize, 1, 1,'single');
            net.params(f).learningRate = LR;
            net.params(f).weightDecay = wd*0.5 ;
            
            f = net.getParamIndex(conv_param_b) ;
            net.params(f).value = zeros(1, 1, 1, 'single');
            net.params(f).learningRate = 2*LR ;
            net.params(f).weightDecay = wd ;
            
            net.addLayer(tanh_layer,dagnn.Tanh(),conv_out,tanh_out);
        else
            net.addLayer(concat_layer,dagnn.Concat(),{rnn_in, gate{ii}}, concat_output);
            %% conv layer
            net.addLayer(conv_layer, ...
                dagnn.Conv('size', [kersize kersize 2 1], 'pad', floor(kersize/2)), ...
                concat_output, conv_out, {conv_param_f,conv_param_b});
            
            f = net.getParamIndex(conv_param_f) ;
            net.params(f).value = conv_f; %1e-3*randn(kersize,kersize,2,1,'single');
            net.params(f).learningRate = LR;
            net.params(f).weightDecay = wd ;
            
            f = net.getParamIndex(conv_param_b) ;
            net.params(f).value = zeros(1, 1, 1, 'single');
            net.params(f).learningRate = 2*LR ;
            net.params(f).weightDecay = wd ;
            
            net.addLayer(tanh_layer,dagnn.Tanh(),conv_out,tanh_out);
        end
        
    else
        net.addLayer(concat_layer,dagnn.Concat(),{rnn_output{ii-1},gate{ii}}, concat_output);
        %% conv layer
        net.addLayer(conv_layer, ...
            dagnn.Conv('size', [kersize kersize 2 1], 'pad', floor(kersize/2)), ...
            concat_output, conv_out, {conv_param_f,conv_param_b});
        
        f = net.getParamIndex(conv_param_f) ;
        net.params(f).value = conv_f;%1e-2*randn(kersize,kersize,2,1,'single') ;
        net.params(f).learningRate = LR;
        net.params(f).weightDecay = wd ;
        
        f = net.getParamIndex(conv_param_b) ;
        net.params(f).value = conv_b ;
        net.params(f).learningRate = 2*LR ;
        net.params(f).weightDecay = wd ;
        
        net.addLayer(tanh_layer,dagnn.Tanh(),conv_out,tanh_out);
    end
    rnn_output{ii}=tanh_out;
end

end