classdef multiply < dagnn.ElementWise
    properties
        method = 'sum';
    end
    %     properties (Transient)
    %         % storing hidden representation in the forward pass
    %         h = [];
    %     end
    
    methods
        function outputs = forward(~, inputs, ~)
            outputs{1}=bsxfun(@times, inputs{1}, inputs{2});
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            derInputs{1}= bsxfun(@times, derOutputs{1}, inputs{2});
            derInputs{2}= bsxfun(@times, derOutputs{1}, inputs{1});
        end
        
        function obj = multiply(varargin)
            obj.load(varargin) ;
        end
    end
end