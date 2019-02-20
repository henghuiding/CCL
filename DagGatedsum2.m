classdef DagGatedsum2 < dagnn.ElementWise
    properties
        method = 'sum';
    end
%     properties (Transient)
%         % storing hidden representation in the forward pass
%         h = [];
%     end
    
    methods
        function outputs = forward(obj, inputs, params)
            if strcmp(obj.method, 'sum')
                [outputs{1}] = Gated_sum_batch2(inputs, params{1});
            else
                [outputs{1}] = Gated_sum_batch2(inputs, params{1});
            end        
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if strcmp(obj.method, 'sum')
                [derInputs,  derParams{1}] = Gated_sum_batch2(inputs, params{1},  derOutputs{1});
            else
                 [derInputs, derParams{1}] = Gated_sum_batch2(inputs, params{1}, derOutputs{1});
            end
        end
        
        function obj = DagGatedsum2(varargin)
            obj.load(varargin) ;
        end
    end
end