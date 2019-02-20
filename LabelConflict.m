classdef LabelConflict < dagnn.ElementWise
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
                [outputs{1}] = LabelConflict_batch(inputs, params{1});
            else
                [outputs{1}] = LabelConflict_batch(inputs, params{1});
            end        
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if strcmp(obj.method, 'sum')
                [derInputs{1},  derParams{1}] = LabelConflict_batch(inputs, params{1},  derOutputs{1});
            else
                 [derInputs{1}, derParams{1}] = LabelConflict_batch(inputs, params{1}, derOutputs{1});
            end
        end
        
        function obj = LabelConflict(varargin)
            obj.load(varargin) ;
        end
    end
end