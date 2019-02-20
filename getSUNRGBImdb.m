function imdb =  getSUNRGBImdb(opts)
% --------------------------------------------------------------------
splitFn = fullfile(opts.dataDir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat');
load(splitFn);

train = trainvalsplit.train;
val = trainvalsplit.val;
test = alltest;

nTrain = numel(train);
nVal = numel(val);
nTest = numel(test);

nImgs = nTrain + nVal + nTest;

fprintf('There are %d training, %d validation and %d testing images.\n',...
    nTrain, nVal, nTest);

%% write image and labels to imdb
data = cell(nImgs,1);
labels = cell(nImgs,1);
sets = [];
iTrain = 0;
iVal = 0;
iTest = 0;
tid = ticStatus('Loading Training Images...',1,1);
for i = 1:nImgs
    
    if i <= nTrain
         imgFolder = train{i};
    elseif i <= nTrain + nVal
         imgFolder = val{i-nTrain};
    else
         imgFolder = test{i-nVal-nTrain};
    end
    
    imgFolder = strrep(imgFolder, '/n/fs/sun3d/data/SUNRGBD/', opts.dataDir);
    imgName = dir([imgFolder, '/image/*.jpg']);
    assert(numel(imgName) == 1);
    
    data{i} = [imgFolder '/image/' imgName.name];
    labels{i} = [imgFolder '/seg_37.png'];
    
    tocStatus(tid,i/nImgs);
end
sets = ones(1, nTrain, 'uint8');
sets = [sets, ones(1, nVal, 'uint8')*2];
sets = [sets, ones(1, nTest, 'uint8')*3];

%% statistics for train images
rgbMean = zeros(1,1,3);
frequency = zeros(1,opts.nClass + 1);
nPixels = 0;
tid = ticStatus('Calculating mean Image...',1,1);
for i = 1 : (nTrain+nVal)
    I = single(imread(data{i}));
    [h,w,~] = size(I);
    nPixels = nPixels + h*w;
    rgbMean = rgbMean + (sum(sum(I,1),2));
    
    label = single(imread(labels{i}));
    frequency = frequency + hist(label(:), 0:opts.nClass);
    tocStatus(tid,i/(nTrain+nVal));
end
rgbMean = rgbMean / nPixels;

%%
imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = sets;
imdb.rgbMean = rgbMean;
imdb.classFrequency = frequency;
imdb.meta.sets = {'train', 'val', 'test'};
