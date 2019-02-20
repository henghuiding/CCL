classdef DagGatedsum_2 < dagnn.ElementWise
    properties
        method = 'sum';
    end
%     properties (Transient)
%         % storing hidden representation in the forward pass
%         h = [];
%     end
    
    methods
        function outputs = forward(obj, inputs, ~)
            if strcmp(obj.method, 'sum')
                [outputs{1}] = Gated_sum_batch_2(inputs);
            else
                [outputs{1}] = Gated_sum_batch_2(inputs);
            end        
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            if strcmp(obj.method, 'sum')
                [derInputs] = Gated_sum_batch_2(inputs,  derOutputs{1});
            else
                 [derInputs] = Gated_sum_batch_2(inputs,  derOutputs{1});
            end
        end
        
        function obj = DagGatedsum_2(varargin)
            obj.load(varargin) ;
        end
    end
end