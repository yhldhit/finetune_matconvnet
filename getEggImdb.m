function imdb = getEggImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
Fea_file = fullfile(opts.dataDir,'FeatureGroupIS0708.mat');
load(Fea_file);
f_index = indexFea2;
labels =  Y2'+1;

unpackPath = fullfile(opts.dataDir, 'BatchIS');
dir_str = dir(fullfile(unpackPath,'*_reg.mat'));

files = {dir_str.name};
files = files(f_index);

%files = [arrayfun(@(n)) sprintf('%d')]
%files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  %{'test_batch.mat'}];
% fullfile the file name
files = cellfun(@(fn) fullfile(unpackPath,fn),files,'UniformOutput',false);
%files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);%文件名加上路径
%file_set = uint8([ones(1, 7), 3*ones(1,3)]);

% if any(cellfun(@(fn) ~exist(fn, 'file'), files))
%   url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
%   fprintf('downloading %s\n', url) ;
%   untar(url, opts.dataDir) ;
% end

% data = cell(1, numel(files));
% labels = cell(1, numel(files));
% sets = cell(1, numel(files));
% for fi = 1:numel(files)
%   fd = load(files{fi}) ;
%   data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
%   labels{fi} = fd.labels' + 1; % Index from 1
%   sets{fi} = repmat(file_set(fi), size(labels{fi}));
% end

% set = cat(2, sets{:});
% data = single(cat(4, data{:}));
data = zeros(224,224,3,numel(labels));
set = uint8([ones(298*7,1);3*ones(numel(labels)-298*7,1)])';
for fi = 1:numel(labels)
    fprintf('*******file index:%d, iband:%d*****\n',fi,opts.iband);
    load(files{fi});
    fd_iband = Im_out(:,:,opts.iband);
    fd_iband = fd_iband.*255;
    fd_iband = imresize(fd_iband,[224,224]);
    fd_iband = repmat(fd_iband,[1,1,3]);
    %fd_iband= imresize(Im_out(:,:,opts.iband),[224,224],'nearest');
    data(:,:,:,fi) = fd_iband;
end
% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],numel(labels)) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 224, 224, 3, []) ;
end
% 
% if opts.whitenData
%   z = reshape(data,[],numel(labels)) ;
%   W = z(:,set == 1)*z(:,set == 1)'/numel(labels) ;
%   [V,D] = eig(W) ;
%   % the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
%   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
%   data = reshape(z, 224, 224, 3, []) ;
% end

% clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = single(data) ;
imdb.images.labels = single(labels);
imdb.images.set = set;
imdb.meta.sets = {'train', 'test'} ;
imdb.meta.averageImage = dataMean;

mkdir(opts.expDir) ;
save(opts.imdbPath, '-struct', 'imdb') ;
% imdb.meta.classes = clNames.label_names;
