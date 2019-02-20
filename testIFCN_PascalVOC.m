function accuracy = testIFCN_PascalVOC()
% run(fullfile(fileparts(mfilename('fullpath')), '../matconvnet-1.0-beta16/matlab/vl_setupnn.m')) ;
% addpath(genpath('../toolbox'));
baseDir = '/home/bshuai/datasets/VOC2012/';

%% General Configuration
gpuDevice(3)
nClass = 21;
imageSize = [448 512 544];
rgbMean = [116.5692 111.4438 102.9367]';
% rgbMean = [116.6725 111.5917 103.1466];
rgbMean = reshape(rgbMean, [1 1 3]) ;
mode = 'val';
flip = true;

opts.eval = true;
opts.save = false;

if strcmp(mode, 'test')
    opts.eval = false;
    opts.save = true;
end

%% location of ground truth
if strcmp(mode, 'val')
    imdb = load(fullfile(baseDir, 'imdb-train-val-update-disk.mat'));
    val = imdb.images.set == 2 & imdb.images.segmentation;
    imgIds = imdb.images.data(val);
    labels = imdb.images.labels(val);
    nImgs = numel(imgIds);
elseif strcmp(mode, 'test')
    imgIds = textread(fullfile(baseDir, 'ImageSets/Segmentation/test.txt'), '%s');
    nImgs = numel(imgIds);
    for i = 1 : nImgs
        imgIds{i} = [baseDir, sprintf('JPEGImages/%s.jpg', imgIds{i})];
    end
end


%% Setting the saved foler (must edit the corresponding name before running)
% ------------------------- Attention -------------------------------------
if opts.save
    
    resultFolder = fullfile(baseDir, 'results/unaryPredictions/IFCN-VOC-valCW(3_8-5_6-7_4)epoch23');
    if ~exist(resultFolder, 'dir'), mkdir(resultFolder); 
    else
        d = dir(resultFolder);
        if numel(d) == nImgs+2
            fprintf('The folder has existed, please check it...\n');
            fprintf('For overriding issue, the programe quit automatically.\n')
            return;
        end
    end
end

%% Information Summary 
fprintf('-------------------- Information summary -------------------------\n');
fprintf('Mode: %s \n', mode);
fprintf('There are %d %s images in total.\n', nImgs, mode);
if opts.save
    fprintf('Results will be saved under the directory:\n');
    fprintf('%s\n', resultFolder);
    fprintf('Stay alert about the possible file override.\n');
end
fprintf('------------------------------------------------------------------\n');

%% testing net
netFn = {};
% dir of net model
netFn{end+1} = sprintf('PascalVOC_%d_COCO/CW(3_8-5_6-7_4)-nh512-batch-ifcn8s-5x5-6layers-1e-3-higher3/net-BN-val.mat', 512); 

fprintf('There are %d nets in total. \n', numel(netFn));
fprintf('------------------------------------------------------------------\n');

nets = loadNet(netFn);
fprintf('Model loading is completed.\n');

%% tesing code
confusion = zeros(nClass);
for i = 1:nImgs  
    
    print = false;
    if i == 1 || mod(i,100) == 0
        print = true;
        fprintf('Labeling testing images %s: %d/%d\n',imgIds{i}, i,nImgs); 
    end
    I = single(imread(imgIds{i}));
    I = bsxfun(@minus, I, rgbMean) ;
    
    sz = [size(I,1), size(I,2)] ;
    
    probs_ = cell(numel(imageSize), 1);
    
    for ss = 1 : numel(imageSize)
    % pertain to training size
    scale = min(imageSize(ss)/sz(1), imageSize(ss)/sz(2)) ;
    
    sz_ = sz * scale;
    sz_ = ceil(sz_ / 32)*32 ;
    I_ = imResample(I, sz_, 'bilinear');
    
    if flip
        I_ = cat(4, I_, fliplr(I_));
    end
    
    I_ = gpuArray(I_);
    
    prob = cell(numel(nets),1);
    
    for jj = 1 : numel(nets)
        net_ = nets{jj};
        input_name = net_.vars(1).name;
        inputs = {input_name, I_};
        net_.eval(inputs) ;

        prob_ = gather(net_.vars(end).value);
        if ~ flip
            prob_ = prob_(:,:,:,1);
        else
            prob_ = (prob_(:,:,:,1) + fliplr(prob_(:,:,:,2))) / 2;
        end
        prob{jj} = prob_;
    end
       
    prob = prob{1} ;                     
    prob = imResample(prob, sz, 'bilinear');   
    probs_{ss} = prob;
    end
    
    prob = probs_{1};
    for ss = 2 : numel(imageSize)
        prob = prob + probs_{ss};
    end
    prob = prob / numel(imageSize);
    [~,pred] = max(prob, [], 3);
    
    
    if opts.eval  
        % ground truth
        gt = imread(labels{i});   
        gt = mod(gt+1, 255);
       % statistics
        ok = gt > 0 ;
        confusion = confusion + accumarray([gt(ok),pred(ok)],1,[nClass nClass]) ;
        [iu, miu, pacc, macc] = getAccuracies(confusion) ;
        if print
            fprintf('IU ') ;
            fprintf('%4.1f ', 100 * iu) ;
            fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
                100*miu, 100*pacc, 100*macc) ;
        end
         
    end
    
    if opts.save
        [~,imgName,~] = fileparts(imgIds{i});
        fn = fullfile(resultFolder, [imgName, '.png']);
        imwrite(pred, labelColors(), fn, 'png');
        
%         [~,imgName,~] = fileparts(imgIds{i});
%         unaryFn = fullfile(resultFolder, [imgName, '.mat']);
%         save(unaryFn, 'prob');
    end  
end

if opts.eval
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * iu) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*miu, 100*pacc, 100*macc) ;
end


end

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;
end

function net = loadNet(netFn)
fprintf('Start loading models.\n');
nNets = numel(netFn);
net = cell(nNets, 1);
for i = 1 : nNets
    fprintf('Loading model %d / %d \n', i, nNets);
    net_ = load(netFn{i}, 'net');
    net_ = net_.net;
    net_ = dagnn.DagNN.loadobj(net_);
    net_.addLayer('prob', ...
        dagnn.SoftMax(), ...
        'prediction', 'probability');
    net_.move('gpu');
    net_.mode = 'test';
    net{i} = net_;
end
end

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
end
