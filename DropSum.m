classdef DropSum < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.
  properties
    rate = 0.5
    frozen = false
  end

  properties (Transient)
    mask
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
        
        obj.numInputs = numel(inputs);
        if strcmp(obj.net.mode, 'test')
            outputs{1} = inputs{1};
            for k = 2 : obj.numInputs
                outputs{1} = outputs{1} + inputs{k} ;
            end
            return ;
        end

        [outputs{1}, obj.mask] = vl_nndropsum(inputs, 'rate', obj.rate) ;
     
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
        if strcmp(obj.net.mode, 'test')
            derInputs = derOutputs ;
            derParams = {} ;
            return ;
        end
        
        derInputs = vl_nndropsum(inputs, derOutputs, 'mask', obj.mask) ;
        derParams = {} ;
        
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = DropSum(varargin)
      obj.load(varargin) ;
    end
    
     function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      obj.mask = [] ;
      obj.frozen = false ;
    end
  end
end
