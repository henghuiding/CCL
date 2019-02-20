function GPU2_updateBN(net, imdb, getBatch, varargin)

opts.expDir = fullfile('data','exp') ;
opts.continue = false ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.derOutputs = {'objective', 1} ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.extractStatsFn = @extractStats ;
opts.mode = 'val';
opts = vl_argparse(opts, varargin) ;

% Initialize (Zeroing out) the BN statistics
moments = [];
 for i = 1:numel(net.layers)
  if isa(net.layers(i).block, 'dagnn.BatchNorm')
    moment = net.getParamIndex(net.layers(i).params{3}) ;
    net.params(moment).value = zeros(size(net.params(moment).value), 'single');
    moments = [moments, moment];
  end
 end
 net.move('gpu');
 opts.moments = moments;
state.getBatch=getBatch;
% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
subset = opts.train ;
opts.nImgs = numel(subset);

fprintf('Start updating BN statistics...\n')






  epoch=1;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
%   state.train = [1:10] ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;
  
  fprintf('Random Check: %d %d %d\n', state.train(1), state.train(100), state.train(end));


  if numGpus <= 1
    process_epoch(net, state, opts, 'train') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      process_epoch(net_, state, opts, 'train') ;
%       if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
%     net = dagnn.DagNN.loadobj(savedNet_{1}) ;
%     stats__ = accumulateStats(stats_) ;
%     stats.train(epoch) = stats__.train ;
%     stats.val(epoch) = stats__.val ;
  end


% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    sate.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.scores = [] ;
% subset = state.(mode) ;
subset = opts.train ;
opts.nImgs = numel(subset);

for t=1:opts.batchSize:numel(subset)
    if t == 1 || mod(t-1, 50) == 0
        fprintf('Processing %3d / %3d batches \n', max(1,fix(t/opts.batchSize)), ...
            ceil(numel(subset)/opts.batchSize));
    end

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(state.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train')
      net.accumulateParamDers = (s ~= 1) ;
      net.eval(inputs, opts.derOutputs) ;
    else
      net.eval(inputs) ;
    end
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    accumulate_moments(net, opts);
  end
end
average_moments(net, opts);

net.move('cpu') ;
net = net.saveobj() ;
modelFn = fullfile(opts.expDir, sprintf('net-BN-%s.mat', opts.mode));
save(modelFn, 'net');













% for t=1:opts.batchSize:numel(subset)
%     
%   if t == 1 || mod(t-1, 50) == 0
%       fprintf('Processing %3d / %3d batches \n', max(1,fix(t/opts.batchSize)), ...
%           ceil(numel(subset)/opts.batchSize));
%   end
% 
%   for s=1:opts.numSubBatches
%     % get this image batch and prefetch the next
%     batchStart = t + (labindex-1) + (s-1) * numlabs ;
%     batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
%     batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
%     if numel(batch) == 0, continue ; end
% 
%     inputs = getBatch(imdb, batch) ;
% 
%     if opts.prefetch
%       if s == opts.numSubBatches
%         batchStart = t + (labindex-1) + opts.batchSize ;
%         batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
%       else
%         batchStart = batchStart + numlabs ;
%       end
%       nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
%       getBatch(imdb, nextBatch) ;
%     end
% 
%       net.accumulateParamDers = (s ~= 1) ;
%       net.eval(inputs, opts.derOutputs) ;
% 
%   end
% 
%     accumulate_moments(net, opts);
% end
% 
% average_moments(net, opts);
% 
% net.move('cpu') ;
% net = net.saveobj() ;
% modelFn = fullfile(opts.expDir, sprintf('net-BN-%s.mat', opts.mode));
% save(modelFn, 'net');

% modelFn = fullfile(opts.expDir, sprintf('net-BN-%s.mat', opts.mode));
% save(modelFn, 'net', '-v7.3');

% -------------------------------------------------------------------------
function accumulate_moments(net, opts)
% -------------------------------------------------------------------------
moments = opts.moments;
for i = 1 : numel(moments)
    jj = moments(i);
    net.params(jj).value = net.params(jj).value + ...
        net.params(jj).der;
end

% -------------------------------------------------------------------------
function average_moments(net, opts)
% -------------------------------------------------------------------------
moments = opts.moments;
for i = 1 : numel(moments)
    jj = moments(i);
    net.params(jj).value = net.params(jj).value / opts.nImgs;
end

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end
