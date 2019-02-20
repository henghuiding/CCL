classdef DagGatedsum < dagnn.ElementWise
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
                [outputs{1}] = Gated_sum_batch(inputs);
            else
                [outputs{1}] = Gated_sum_batch(inputs);
            end        
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams=[];
            if strcmp(obj.method, 'sum')
                [derInputs] = Gated_sum_batch(inputs,  derOutputs{1});
            else
                 [derInputs] = Gated_sum_batch(inputs,  derOutputs{1});
            end
        end
        
        function obj = DagGatedsum(varargin)
            obj.load(varargin) ;
        end
    end
end