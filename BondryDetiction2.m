classdef BondryDetiction2 < dagnn.ElementWise
    properties
        method = 'max';
    end
    %     properties (Transient)
    %         % storing hidden representation in the forward pass
    %         h = [];
    %     end
    
    methods
        function outputs = forward(~, inputs, ~)
            inputs{1}(inputs{1}~=0)=0.8;
%             inputs{1}(inputs{1}~=0)=0.4;
            outputs{1}=inputs{1}+0.2;
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            derInputs{1}= 0*inputs{1};
        end
        
        function obj = BondryDetiction2(varargin)
            obj.load(varargin) ;
        end
    end
end