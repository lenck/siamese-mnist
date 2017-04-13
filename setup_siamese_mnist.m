function setup_siamese_mnist()
%SETUP_SIAMESE_MNIST Sets up siamese-mnist, by adding its folders to the Matlab path
root = fileparts(mfilename('fullpath')) ;
addpath(root, fullfile(root, 'matlab')) ;
end

