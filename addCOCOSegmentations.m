function imdb = addCOCOSegmentations(imdb)
% --------------------------------------------------------------------
addpath(genpath('~/codes/toolbox'));

dataDir = '/home/bshuai/datasets/COCO/' ;
%% training images
imgDir = fullfile(dataDir, 'images/train2014/');
gtDir = fullfile(dataDir, 'annotations/train/');
imgIds = dir([gtDir '*.png']); 
imgIds = {imgIds.name};
nImgs = length(imgIds); 
for i = 1:nImgs, 
    imgIds{i} = imgIds{i}(1:end-4); 
end
fprintf('There are total %d train images.\n',nImgs);

nTrainImgs = 0 ;
data = cell(1, nImgs);
labels = cell(1, nImgs);
tid = ticStatus('Loading Training Images...',1,1);
for i = 1:nImgs
    if i == 19051
        continue; % corrupted image, skip it
    end
    nTrainImgs = nTrainImgs + 1;
    data{nTrainImgs} = ([imgDir imgIds{i} '.jpg']);
    labels{nTrainImgs} = ([gtDir imgIds{i} '.png']);
    tocStatus(tid,i/nImgs);
end
sets = ones(1,nTrainImgs, 'uint8');
data = data(1:nTrainImgs);
labels = labels(1:nTrainImgs);

%% validation images
% imgDir = fullfile(dataDir, 'images/validation/');
% gtDir = fullfile(dataDir, 'annotations/validation/');
% imgIds = dir([gtDir '*.png']); 
% imgIds = {imgIds.name};
% nImgs = length(imgIds); 
% for i = 1:nImgs, 
%     imgIds{i} = imgIds{i}(1:end-4); 
% end
% fprintf('There are total %d validation images.\n',nImgs);
% 
% data = [data; cell(nImgs,1)];
% labels = [labels; cell(nImgs,1)];
% for i = 1:nImgs
%     data{nTrainImgs+i} = [imgDir imgIds{i} '.jpg'];
%     labels{nTrainImgs+i} = [gtDir imgIds{i} '.png'];
% end
% sets = [sets ones(1,nImgs, 'uint8')];
% nValImgs = nImgs;

%% Update the imdb structure
imdb.images.data = [imdb.images.data, data] ;
imdb.images.labels = [imdb.images.labels, labels] ;
imdb.images.set = [imdb.images.set, sets];
imdb.images.segmentation = [imdb.images.segmentation, ...
    true(1, nTrainImgs)] ;

imdb.images = rmfield(imdb.images, 'id');
imdb.images = rmfield(imdb.images, 'name');
imdb.images = rmfield(imdb.images, 'classification');
imdb.images = rmfield(imdb.images, 'size');

classes = imdb.classes;
classFrequency = imdb.classFrequency;
images = imdb.images;
rgbMean = imdb.rgbMean;
sets = imdb.sets;
meta = imdb.meta;

save('/home/bshuai/datasets/VOC2012/imdb-train-val-coco-disk.mat', ...
    'classes', 'classFrequency', 'images', 'rgbMean', 'sets', 'meta');

