function acc_val = test_val_data(net_path,iband)
%net = load ('D:\eggproject\qilong\batch3\expdata\net-epoch-50-deployed.mat');
net = load(net_path);
run matlab\vl_setupnn;
% opts.dataDir = 'D:\eggproject\qilong\batch3\' ;
% opts.expDir  = 'D:\eggproject\qilong\batch3\expdata';
% % µ¼ÈëÔ¤ÑµÁ·µÄmodel
% opts.modelPath = fullfile('D:\eggproject\qilong\model','imagenet-vgg-verydeep-19.mat');
% % [opts, varargin] = vl_argparse(opts, varargin) ;
% 
% %opts.numFetchThreads = 12 ;
% opts.iband = 1;
% opts.lite = false ;
% imdbname = ['imdb-' num2str(opts.iband) '.mat'];
% opts.imdbPath = fullfile(opts.expDir, imdbname);
% 
% opts.train = struct() ;
% opts.train.gpus = [1];
% opts.train.batchSize = 100;
% opts.train.numSubBatches = 4 ;
% opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];
% %opts = vl_argparse(opts, varargin) ;
% opts.contrastNormalization = true;
% opts.batchSize = 256 ;
% opts.numSubBatches = 1 ;
% opts.train = [] ;
% opts.val = [] ;
% opts.numEpochs = 50 ;
% opts.gpus = [1] ; % which GPU devices to use (none, one, or more)
% opts.learningRate = 0.001 ;
% opts.continue = false ;
% opts.expDir = fullfile('data','exp') ;
% opts.conserveMemory = false ;
% opts.backPropDepth = +inf ;
% opts.sync = false ;
% opts.prefetch = false ;
% opts.weightDecay = 0.0005 ;
% opts.momentum = 0.9 ;
% opts.errorFunction = 'binary' ;
% opts.errorLabels = {} ;
% opts.plotDiagnostics = false ;
% opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
%opts = vl_argparse(opts,['expDir', opts.expDir,opts.train]) ;

% net = cnn_deploy(net);
% modelPath = 'D:\eggproject\qilong\batch3\expdata\net-epoch-50-deployed.mat';
% save(modelPath, '-struct', 'net') ;
net = vl_simplenn_move(net, 'gpu');
imdbPath = ['E:\eggproject\batch3\expdata\data\imdb-' num2str(iband)];
load(imdbPath);
valset = find(images.set == 3);
dataval = images.data(:,:,:,valset);
labelval = images.labels(valset);
labelpsudo = [];
dataval = gpuArray(dataval);
for i = 1:size(dataval,4)
    data_ = dataval(:,:,:,i);
    res = vl_simplenn(net,data_);
    scores = squeeze(gather(res(end).x));
    [bestScore,best] = max(scores);
    labelpsudo(end+1) = best;
end

acc_val = length(find(labelval == labelpsudo))*1.0/length(labelval);
fprintf('*****************accuracy on band %d is %f**********\n',iband,acc_val);
