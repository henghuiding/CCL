classdef Confidence < dagnn.ElementWise
%     properties
%         method = 'sum';
%     end
%     properties (Transient)
%         % storing hidden representation in the forward pass
%         h = [];
%     end
    
    methods
        function outputs = forward(~, inputs,~)
                [outputs{1}] = Confidence_batch(inputs);     
        end
        
        function [derInputs,derParams] = backward(~, inputs,~,derOutputs)
%                 [derInputs] = Confidence_batch(inputs,derOutputs{1});
                derInputs={};
                derParams={};

        end        
        function obj = Confidence(varargin)
            obj.load(varargin) ;
        end
    end
end