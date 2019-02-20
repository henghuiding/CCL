function newAutotestIFCN_PascalContext(networkfolder)
% run(fullfile(fileparts(mfilename('fullpath')), '../matconvnet-1.0-beta16/matlab/vl_setupnn.m')) ;
% addpath(genpath('../toolbox'));
addpath(genpath('utils'));
baseDir = '/home/hhding/datasets/VOC2010/';

%% General Configuration
% gpuDevice(3)
nClass = 59;
imageSize = 512;
rgbMean = [116.6268 111.9239 103.6421]';
rgbMean = reshape(rgbMean, [1 1 3]) ;
mode = 'val';
flip = true;

eval = true;
save = false;

sigma1=[50, 10, 10, 5, 5];
sigma2=[50, 5, 5, 5];
%% location of ground truth
if strcmp(mode, 'val')
    imgDir = fullfile(baseDir, 'Images/');
    mskDir = fullfile(baseDir, 'Masks10-30/Masks15/');
    gtDir = fullfile(baseDir, 'LabelsSemantic-59/');
elseif strcmp(mode, 'test')
    eval = false;
    save = true;
    imgDir = fullfile(baseDir, 'release_test/testing/');
else
    error('Unrecognized mode...');
end

% % Split file
splitFn = fullfile(baseDir, 'VOC-testId.mat');
load(splitFn);
testId = sets == 3;

imgIds=dir([gtDir '*.mat']);
imgIds={imgIds(testId).name};
nImgs=length(imgIds);
for i=1:nImgs,
    imgIds{i}=imgIds{i}(1:end-4);
end

%%
[~,~,classes] = textread(fullfile(baseDir, '59_labels.txt'), '%d %s %s', nClass);
cmap = labelColors(nClass+1);


%% Setting the saved foler (must edit the corresponding name before running)
% ------------------------- Attention -------------------------------------
if save
    
    resultFolder = fullfile(baseDir, 'IFCN-VGG16-predictions');
    if ~exist(resultFolder, 'dir'), mkdir(resultFolder); end
end

%% Information Summary
fprintf('-------------------- Information summary -------------------------\n');
fprintf('Mode: %s \n', mode);
fprintf('There are %d %s images in total.\n', nImgs, mode);
if save
    fprintf('Results will be saved under the directory:\n');
    fprintf('%s\n', resultFolder);
    fprintf('Stay alert about the possible file override.\n');
end
fprintf('------------------------------------------------------------------\n');

%% testing net
netFn = {};
resNet = {};

netFn{end+1} = sprintf('%s/net-BN-val.mat', networkfolder);

fprintf('There are %d nets in total. \n', numel(netFn));
fprintf('------------------------------------------------------------------\n');

nets = loadNet(netFn);
fprintf('Model loading is completed.\n');

%% tesing code
confusion = zeros(nClass);
consumedTime = 0;
for i = 1:nImgs
    
    print = false;
    if i == 1 || mod(i,500) == 0
        print = true;
        fprintf('Labeling testing images %s: %d/%d\n',imgIds{i}, i,nImgs);
    end
    I0 = single(imread([imgDir imgIds{i} '.jpg']));
    I = bsxfun(@minus, I0, rgbMean) ;
    
    msk = load([mskDir imgIds{i} '.mat']);
    msk = single(msk.msk);
    
    sz = [size(I,1), size(I,2)] ;
    
    % pertain to training size
    scale = min(imageSize/sz(1), imageSize/sz(2)) ;
    
    sz_ = sz * scale;
    sz_ = ceil(sz_ / 32)*32 ;
    newscale = max(sz(1)/sz_(1), sz(2)/sz_(2)) ;
    sx = round(newscale * ((1:sz_(1)) - sz_(1)/2) + sz(1)/2) ;
    sy = round(newscale * ((1:sz_(2)) - sz_(2)/2) + sz(2)/2) ;
    okx = find(1 <= sx & sx <= sz(1)) ;
    oky = find(1 <= sy & sy <= sz(2)) ;
    I_=zeros(sz_(1),sz_(2),3, 'single');
    I_(okx,oky,:) = I(sx(okx),sy(oky),:);
    
    lmx1 = 1 : 8 : sz_(1); lmy1 = 1 : 8 : sz_(2);
    lmx2 = 1 : 4 : sz_(1); lmy2 = 1 : 4 : sz_(2);
    %     I_ = imResample(I, sz_, 'bilinear');
    mask_org= ones(sz_, 'single') ;
    mask_org(okx,oky) = msk(sx(okx),sy(oky));
    mask_org(mask_org>1)=4;
    mask8=zeros(sz_/8,'single');mask4=zeros(sz_/4,'single');
    
    gausFilter = fspecial('gaussian', [100,100], sigma1(1));
    gaus = imfilter(mask_org, gausFilter, 'replicate');
    mask8(:,:,1) = gaus (lmx1,lmy1); %for protect high level prediction
    mask4(:,:,1) = gaus (lmx2,lmy2); %for protect high level prediction
    mask_org(mask_org==1)=0.8;
    for ii=2:5
        gausFilter = fspecial('gaussian', [30,30], sigma1(ii));
        gaus = imfilter(mask_org, gausFilter, 'replicate');
        mask8(:,:,ii) = gaus (lmx1,lmy1);
    end
    mask_org(mask_org==1)=0.5;
    for ii=2:4
            gausFilter = fspecial('gaussian', [30,30], sigma2(ii));
            gaus = imfilter(mask_org, gausFilter, 'replicate');
            mask4(:,:,ii) = gaus(lmx2,lmy2) * 0.8;
    end
    if flip
        I_ = cat(4, I_, fliplr(I_));
        mask8 = cat(4, mask8, fliplr(mask8));
        mask4 = cat(4, mask4, fliplr(mask4));
    end
    
    I_ = gpuArray(I_);
    mask8=gpuArray(mask8);
    mask4=gpuArray(mask4);
    
    prob = cell(numel(nets),1);
    
    for jj = 1 : numel(nets)
        net_ = nets{jj};
        input_name = net_.vars(1).name;
        inputs = {input_name, I_, 'mask8', mask8, 'mask4', mask4};
        
        tic;
        
        net_.eval(inputs) ;
        
        consumedTime = consumedTime + toc;
        
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
    [~,pred] = max(prob, [], 3);
    
    
    if eval
        % ground truth
        gt = load([gtDir imgIds{i} '.mat']);
        gt = gt.LabelMap;
        % statistics
        ok = gt > 0 ;
        confusion = confusion + accumarray([gt(ok),pred(ok)],1,[nClass nClass]) ;
        [iu, ac, miu, pacc, macc] = getAccuracies(confusion) ;
        if print
            fprintf(' IU ') ;
            fprintf('%4.2f ', 100 * iu) ;
            fprintf('\n AC ') ;
            fprintf('%4.2f ', 100 * ac) ;
            fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
                100*miu, 100*pacc, 100*macc) ;
        end
        
    end
    
    if save
        draw_label_image( uint8(I0), pred, gt, cmap, ['unlabeled';classes]);
        fn = fullfile(resultFolder, [imgIds{i}, '.png']);
        
        h = gca;
        F = getframe(h);
        im = F.cdata;
        imwrite(im, fn);
        %         print(fn,'-dpng','-r96');
        %         close all;
    end
end

if eval
    fprintf(' IU ') ;
    fprintf('%4.2f ', 100 * iu) ;
    fprintf('\n AC ') ;
    fprintf('%4.2f ', 100 * ac) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*miu, 100*pacc, 100*macc) ;
end

fprintf('Total consumed time %5.2f, Average consumed time per image %5.2f\n', consumedTime*1000, consumedTime / nImgs * 1000);


end

% -------------------------------------------------------------------------
function [IU, AC, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
AC = tp ./ max(1, pos) ;
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
