function [net, info, imdb] = cnn_mnist_siamese(varargin)
% CNN_MNIST_SIAMESE  Demonstrated MatConNet on MNIST Siamese network

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(vl_rootnn, 'examples', 'mnist'));

opts.batchNormalization = false ;
opts.network = [] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = 'siam-dagnn' ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.seed = 1;
opts.train = struct('gpus', [], 'numEpochs', 10);
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
    'networkType', 'dagnn') ;
  net.removeLayer({'layer7', 'layer8', 'top1err', 'top5err'});
  net.addLayer('emb', ...
    dagnn.Conv('size', [1 1 500 2], 'stride', 1, 'pad', 0), ...
    'x6', 'emb', {'emb_f', 'emb_b'}) ;
  net.initParams(); net.rebuild();
  net = vl_create_siamese(net, net, 'mergeParams', true);
  net.addLayer('loss', dagnn.ContrastiveLoss(), ...
    [net.getOutputs(), 'label'], {'objective'}, {});
else
  net = opts.network ;
  opts.network = [] ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_mnist_setup_data(opts);
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% The training is quite sensitive to data scale
scale = 0.00390625;
imdb.images.data = imdb.images.data * scale;

q = RandStream('mt19937ar','Seed',opts.seed);
trainSel = find(imdb.images.set == 1);
trainSel = [trainSel; randsample(q, trainSel, numel(trainSel))];
valSel = find(imdb.images.set == 3);
valSel = [valSel; randsample(q, valSel, numel(valSel))];
pairs = [trainSel, valSel];
pairsSet = [ones(1, size(trainSel, 2)), 2*ones(1, size(valSel, 2))];
getLabel = @(sel) imdb.images.labels(pairs(1,sel)) == imdb.images.labels(pairs(2,sel));
getImagesA = @(sel) imdb.images.data(:,:,:,pairs(1, sel)) ;
getImagesB = @(sel) imdb.images.data(:,:,:,pairs(2, sel)) ;

net.meta.classes.name = ...
  arrayfun(@(x)sprintf('%d',x), 1:10, 'UniformOutput', false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, build_get_batch(opts), ...
  'expDir', opts.expDir, net.meta.trainOpts, opts.train, ...
  'train', find(pairsSet==1), 'val', find(pairsSet==2)) ;

% --------------------------------------------------------------------
function fn = build_get_batch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) get_siamese_batch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = get_siamese_batch(opts, imdb, batch)
% --------------------------------------------------------------------
imagesA = getImagesA(batch);
imagesB = getImagesB(batch);
labels = getLabel(batch);
if opts.numGpus > 0
  imagesA = gpuArray(imagesA) ;
  imagesB = gpuArray(imagesB) ;
end
inputs = {'input_a', imagesA, 'input_b', imagesB, 'label', labels} ;
end
end
