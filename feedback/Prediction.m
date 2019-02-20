classdef Prediction < dagnn.ElementWise
    properties
        method = 'max';
    end
    %     properties (Transient)
    %         % storing hidden representation in the forward pass
    %         h = [];
    %     end
    
    methods
        function outputs = forward(~, inputs, ~)
            [~,outputs{1}] = max(inputs{1}, [], 3);
        end
        
        function [derInputs,derParams] = backward(~, inputs, ~,derOutputs)
            derParams={};
            derInputs{1}= 0*inputs{1};
        end
        
        function obj = Prediction(varargin)
            obj.load(varargin) ;
        end
    end
end