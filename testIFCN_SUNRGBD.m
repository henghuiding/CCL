function accuracy = testIFCN_SUNRGBD()
run(fullfile(fileparts(mfilename('fullpath')), '../matconvnet-1.0-beta16/matlab/vl_setupnn.m')) ;
addpath(genpath('../toolbox'));
addpath(genpath('utils'));
baseDir = '/home/bshuai/datasets/SUNRGBD/';

%% General Configuration
gpuDevice(3)
nClass = 37;
imageSize = 512 ;
% rgbMean = [116.5692 111.4438 102.9367]';
rgbMean = [127.0942 118.9957 112.7816]';
rgbMean = reshape(rgbMean, [1 1 3]) ;
mode = 'test';
flip = true;

opts.eval = true;
opts.save = false;

% if strcmp(mode, 'test')
%     opts.eval = false;
%     opts.save = true;
% end

%% location of ground truth

imdb = load(fullfile(baseDir, 'imdb-train-val-disk.mat'));
test = imdb.images.set == 3 ;
imgIds = imdb.images.data(test);
labels = imdb.images.labels(test);
nImgs = numel(imgIds);

%% Class Names
load(fullfile(baseDir, 'SUNRGBDtoolbox/Metadata/seg37list.mat'), 'seg37list');
classes = seg37list;
cmap = labelColors(nClass+1);

% replace underline in class name list
for i = 1 : nClass
    classes{i} = strrep(classes{i}, '_', ' ');
end

%% Setting the saved foler (must edit the corresponding name before running)
% ------------------------- Attention -------------------------------------
if opts.save   
    resultFolder = fullfile(baseDir, 'IFCN-VGG16-predictions');
    if ~exist(resultFolder, 'dir'), mkdir(resultFolder); end
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

netFn{end+1} = sprintf('SUNRGBD_%d_TrainVal/batch-ifcn8s-5x5-6layers-1e-3-higher3/net-BN-test.mat', 512); 

fprintf('There are %d nets in total. \n', numel(netFn));
fprintf('------------------------------------------------------------------\n');

nets = loadNet(netFn);
fprintf('Model loading is completed.\n');

%% tesing code
confusion = zeros(nClass);
cls_fre = zeros(1, nClass+1);
for i = 1:nImgs  
    
    print = false;
    if i == 1 || mod(i,100) == 0
        print = true;
        fprintf('Labeling testing images %s: %d/%d\n',imgIds{i}, i,nImgs); 
    end
    I0 = single(imread(imgIds{i}));
    I = bsxfun(@minus, I0, rgbMean) ;
    
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
    [~,pred] = max(prob, [], 3);
    
    
    if opts.eval  
        % ground truth
        gt = imread(labels{i});   
        
        cls_fre = cls_fre + hist(gt(:), 0:nClass);

       % statistics
        ok = gt > 0 ;
        confusion = confusion + accumarray([gt(ok),pred(ok)],1,[nClass nClass]) ;
        [iu, ac, miu, pacc, macc] = getAccuracies(confusion) ;
        if print
            fprintf('IU ') ;
            fprintf('%4.2f ', 100 * iu) ;
            fprintf('\nAC ') ;
            fprintf('%4.2f ', 100 * ac) ;
            fprintf('\n meanAC: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
                100*miu, 100*pacc, 100*macc) ;
        end
         
    end
    
    if opts.save
        draw_label_image( uint8(I0), pred, gt, cmap, ['unlabeled' classes]);
        fn = fullfile(resultFolder, [num2str(i, '%.4d'), '.png']);
      
        h = gca;
        F = getframe(h);
        im = F.cdata;
        imwrite(im, fn);

    end  
end

realnClass = sum(cls_fre(2:end) > 0);

if opts.eval
    fprintf('IU ') ;
    fprintf('%4.2f ', 100 * iu) ;
    fprintf('\nAC ') ;
    fprintf('%4.2f ', 100 * ac) ;
    fprintf('\n meanAC: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*miu, 100*pacc, 100*macc) ;
end


end

% -------------------------------------------------------------------------
function [IU, AC, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
AC = tp ./ max(1, pos);
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(AC) ;
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

