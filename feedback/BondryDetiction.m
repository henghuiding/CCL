classdef BondryDetiction < dagnn.ElementWise
    properties
        method = 'gaussian';
        size = 27;
        sigma = 5;
        value = 2;
    end
    %     properties (Transient)
    %         % storing hidden representation in the forward pass
    %         h = [];
    %     end
    
    methods
        function outputs = forward(obj, inputs, ~)
            inputs{1}(inputs{1}~=0)= obj.value-0.5;%a=gather(outputs{1}(:,:,:,1));
            inputs{1} = inputs{1} + 0.5;
            gausFilter = fspecial('gaussian', [obj.size,obj.size], obj.sigma);%b=gather(inputs{1}(:,:,:,1));
            outputs{1} = imfilter(inputs{1}, gausFilter, 'replicate');
%             outputs{1} = inputs{1};
        end
        
        function [derInputs,derParams] = backward(obj, inputs, ~,derOutputs)
            derParams={};
            derInputs{1}= 0*inputs{1};
        end
        
        function obj = BondryDetiction(varargin)
            obj.load(varargin) ;
        end
    end
end