function imdb = getImageNetImdb_disk(opts)
% --------------------------------------------------------------------
%% training images
imgDir = fullfile(opts.dataDir, 'images/training/');
gtDir = fullfile(opts.dataDir, 'annotations/training/');
imgIds = dir([imgDir '*.jpg']); 
imgIds = {imgIds.name};
nImgs = length(imgIds); 
for i = 1:nImgs, 
    imgIds{i} = imgIds{i}(1:end-4); 
end
fprintf('There are total %d train/ images.\n',nImgs);
assert(nImgs == 20210);

data = cell(nImgs,1);
labels = cell(nImgs,1);
tid = ticStatus('Loading Training Images...',1,1);
for i = 1:nImgs
    data{i} = ([imgDir imgIds{i} '.jpg']);
    labels{i} = ([gtDir imgIds{i} '.png']);
    tocStatus(tid,i/nImgs);
end
sets = ones(1,nImgs, 'uint8');

nTrainImgs = nImgs;

%% validation images
imgDir = fullfile(opts.dataDir, 'images/validation/');
gtDir = fullfile(opts.dataDir, 'annotations/validation/');
imgIds = dir([imgDir '*.jpg']); 
imgIds = {imgIds.name};
nImgs = length(imgIds); 
for i = 1:nImgs, 
    imgIds{i} = imgIds{i}(1:end-4); 
end
fprintf('There are total %d validation images.\n',nImgs);
assert(nImgs == 2000);

data = [data; cell(nImgs,1)];
labels = [labels; cell(nImgs,1)];
for i = 1:nImgs
    data{nTrainImgs+i} = [imgDir imgIds{i} '.jpg'];
    labels{nTrainImgs+i} = [gtDir imgIds{i} '.png'];
end
sets = [sets ones(1,nImgs, 'uint8')*2];

nImgs = nTrainImgs + nImgs;
assert(nImgs == numel(data));
nTrainValImgs = nImgs;

%% test images
imgDir = fullfile(opts.dataDir, 'release_test/testing/');
imgIds = dir([imgDir '*.jpg']); 
imgIds = {imgIds.name};
nImgs = length(imgIds); 
for i = 1:nImgs, 
    imgIds{i} = imgIds{i}(1:end-4); 
end
fprintf('There are total %d testing images.\n',nImgs);
assert(nImgs == 3352);

data = [data; cell(nImgs,1)];
for i = 1:nImgs
    data{nTrainValImgs+i} = [imgDir imgIds{i} '.jpg'];
end
sets = [sets ones(1,nImgs, 'uint8')*3];

nImgs = nTrainValImgs + nImgs;
assert(nImgs == numel(data));

%% statistics for train images
tid = ticStatus('Calculating mean Image...',1,1);
rgbMean = zeros(1,1,3);
frequency = zeros(1,151);
nPixels = 0;
for i = 1 : nTrainValImgs
    I = imread(data{i});
    [h,w,~] = size(I);
    nPixels = nPixels + h*w;
    rgbMean = rgbMean + (sum(sum(I,1),2));
    
    label = imread(labels{i});
    frequency = frequency + hist(label(:), 0:150);
    tocStatus(tid,i/nTrainValImgs);
end
rgbMean = rgbMean / nPixels;

%%
imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = sets;
imdb.rgbMean = rgbMean;
imdb.classFrequency = frequency;
imdb.meta.sets = {'train', 'val', 'test'};