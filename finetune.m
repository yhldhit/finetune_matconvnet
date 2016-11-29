function [net, info] = finetune(varargin)
%%
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
% 修改读入文件夹的路径
opts.dataDir = 'D:\eggproject\qilong\batch3\' ;
opts.expDir  = 'E:\eggproject\batch3\expdata\data';
% 导入预训练的model
opts.modelPath = fullfile('D:\eggproject\qilong\model','imagenet-vgg-verydeep-19.mat');
%[opts, varargin] = vl_argparse(opts, varargin) ;
opts.train = struct() ;
opts.train.gpus = [1];
opts.train.batchSize = 40;
opts.train.numSubBatches = 4 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];
opts.contrastNormalization = true;
opts.lite = false ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
opts = vl_argparse(opts, varargin) ;
acc_list = [];
%opts.numFetchThreads = 12 ;
for iband = 141:171
    fprintf('*******************index band: %d****************\n',iband);
    modelPath = ['E:\eggproject\batch3\expdata\model\net-iband-' num2str(iband) '.mat'];

    opts.iband = iband;
    imdbname = ['imdb-' num2str(opts.iband) '.mat'];
    opts.imdbPath = fullfile(opts.expDir, imdbname);
    if exist(modelPath,'file')
        acc_list(end+1) = test_val_data(modelPath,iband);
        continue;
    end
    %------------------------------
    %prepare model
    %------------------------------
    %load net
    net = load(opts.modelPath);
    %adjust net
    net.layers  = net.layers(1:end-2);
    %set learning rate
    lr = [0.005 0.002];
    for i = 1:size(net.layers,2)
        if(strcmp(net.layers{i}.type,'conv'))
            net.layers{i}.learningRate = lr;
        end
    end
    net.layers{end+1} = struct('name','fc8_egg','type', 'conv', ...
    'weights', {{0.05*randn(1,1,4096,2, 'single'), zeros(1,2,'single')}}, ...
    'learningRate',[0.005 0.002], ...
    'stride', [1 1], ...
    'pad', [0 0 0 0], ...
    'opts',{cell(0,0)}) ;
    net.layers{end+1} = struct('type','softmaxloss','opts',{cell(0,0)});
    %-------------------------
    %prepare data
    %-------------------------
    %delete(opts.imdbPath);
    if exist(opts.imdbPath,'file')
      imdb = load(opts.imdbPath) ;
%       imdb.images.labels = imdb.images.labels-1;
    else
      imdb = getEggImdb(opts) ;
%       mkdir(opts.expDir) ;
%       save(opts.imdbPath, '-struct', 'imdb') ;
    end

    % -------------------------------------------------------------------------
    %                                                                     Learn
    % -------------------------------------------------------------------------
    opts.train.train = find(imdb.images.set==1) ;%训练集
    opts.train.val = find(imdb.images.set==3) ;%测试集
    [net, info] = cnn_train(net, imdb,opts.iband, @getBatch, ...
                          'expDir', opts.expDir, ...
                          opts.train);
    %------------------------------------------
    %                               deploy
    %------------------------------------------
    net = cnn_deploy(net);
    save(modelPath, '-struct', 'net') ; 
    acc_list(end+1) = test_val_data(modelPath,iband);
end 
resultpath = ['E:\eggproject\batch3\expdata\results\' 'acc_list-' num2str(iband)];
save(resultpath,'acc_list');
% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, im=fliplr(im) ; end         

