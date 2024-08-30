%% Dependencies
clear all;close all;
log_file = {};
addpath(genpath('./Utils/'))
%% load shapes to match
data_folder = '../data/test/processed/';
names = ["test_scan_006.mat" ...
        "test_scan_073.mat" ...
        "tr_scan_080.mat" ...
        "tr_scan_081.mat"...
        "result_generic_csr0128a_step2.mat" ...
        "result_generic_csr0268a_step2.mat"
];
for j = 1:6
        fig = subplot(3, 2, j);
        shape = load([data_folder convertStringsToChars(names(j))]).X;
        shape.TRIV = shape.triv;
        shape.VERT = shape.vert;
        % change this line to visualize different landmarks
        i = '0';
        color = load(['res/faust_scan_remeshed/res_' i '_' convertStringsToChars(names(j))]);
        color_sorted = sort(color.p, 'ascend');
        colors = zeros(size(shape.vert));
        colors(:,3)= 1;
        colors(:,2) = rescale(color.p);
        colors(:,1) = rescale(color.p);
        colormap(colors);
        plot_scalar_map(shape, [1: size(shape.VERT,1)]');
        freeze_colors;title(names(j));
end

%%
