function imdb = getPascalContextImdb_disk(opts)
% --------------------------------------------------------------------
imgDir = fullfile(opts.dataDir, 'Images/');
gtDir = fullfile(opts.dataDir, 'LabelsSemantic-59/');
splitFn = fullfile(opts.dataDir, 'VOC-testId.mat');
imgIds = dir([gtDir '*.mat']); 
imgIds = {imgIds.name};
nImgs = length(imgIds); 
for i = 1:nImgs, 
    imgIds{i} = imgIds{i}(1:end-4); 
end
fprintf('There are total %d train/val images.\n',nImgs);
assert(nImgs == 10103);

data = cell(nImgs,1);
labels = cell(nImgs,1);
tid = ticStatus('Loading Training Images...',1,1);
for i = 1:nImgs
    data{i} = ([imgDir imgIds{i} '.jpg']);
    labels{i} = ([gtDir imgIds{i} '.mat']);
    tocStatus(tid,i/nImgs);
end
sets = load(splitFn, 'sets');
sets = uint8(sets.sets);
assert(numel(sets)==nImgs);


nTrainImgs = sum(sets == 1);
train_data = data(sets == 1);
train_label = labels(sets == 1);
%% statistics for train images
tid = ticStatus('Calculating mean Image...',1,1);
rgbMean = zeros(1,1,3);
frequency = zeros(1,60);
nPixels = 0;
for i = 1 : nTrainImgs
    I = single(imread(train_data{i}));
    [h,w,~] = size(I);
    nPixels = nPixels + h*w;
    rgbMean = rgbMean + (sum(sum(I,1),2));
    
    label = load(train_label{i});
    label = label.LabelMap;
    frequency = frequency + hist(label(:), 0:59);
    tocStatus(tid,i/nTrainImgs);
end
rgbMean = rgbMean / nPixels;

%%
imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = sets;
imdb.rgbMean = rgbMean;
imdb.classFrequency = frequency;
imdb.meta.sets = {'train', 'val', 'test'};