addpath tools/functions
clear all;clc;close all

%% Preconditioners paths
DPWFS_path = "../Pytorch/training_results/Paper/06-07-2023/r1/DE/";
DPWFSn_path = "../Pytorch/training_results/Paper/06-07-2023/n1/DE/";

FigurePath = "./figures/paper/Figure6/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "Supplemental I.mp4";
FigureNameB = "Fig_5.pdf";

DE_list = dir(DPWFS_path+"*.mat");
DE_list = {DE_list.name};
DE_list = natsort(DE_list);

DEn_list = dir(DPWFSn_path+"*.mat");
DEn_list = {DEn_list.name};
DEn_list = natsort(DEn_list);

% stack
DE = ones(512,512,1);DEn = ones(512,512,1);
for k = 1:length(DE_list)-1
   load(DPWFS_path+DE_list{k})
   DE = cat(3,DE,fftshift(OL1));
   load(DPWFSn_path+DEn_list{k})
   DEn = cat(3,DEn,fftshift(OL1));
end

%% Prepare plot

fig = figure('Color','w',Units='normalized',Position=[0.2844 0.2574 0.5464 0.5000]);

ha = tight_subplot(1,2,[.02 .02],[.019 .02],[.02 .02]);
axes(ha(1))
im1 = imagesc(DE(:,:,2),[-pi pi]); colormap jet;c1 = colorbar;axis image
axis off
tx1 = annotation('textbox',[0.1 0.75 0.3 0.15],"String","DPWFS-R1",'FontSize',20);
tx1.LineStyle = 'none';tx1.FontWeight = 'bold';
c1.Visible = 'off';
ha(1).Position = [0.095 0.02 0.3982 0.96];

axes(ha(2))
im2 = imagesc(DEn(:,:,2),[-pi pi]); colormap jet;c2 = colorbar;axis image
axis off
tx1 = annotation('textbox',[0.52 0.75 0.3 0.15],"String","DPWFS-N1",'FontSize',20);
tx1.LineStyle = 'none';tx1.FontWeight = 'bold';
c2.Ticks = [-pi 0 pi];
c2.TickLabels = {"-\pi","0","\pi"};
c2.FontSize = 24;

stx = sgtitle('Epoch 0','FontSize',22);

%%
writer = VideoWriter(FigurePath+FigureNameA, 'MPEG-4');
writer.FrameRate = 30;
frames = 1000; % Number of frames
ff = 1:frames;
frame_idx = round((ff.^2.4)/(frames.^2.4)*99+1);

open(writer);
% Generate the animation
for fr = 1:frames
    im1.CData = DE(:,:,frame_idx(fr));
    im2.CData = DEn(:,:,frame_idx(fr));
    stx.String = "Epoch "+frame_idx(fr);
    drawnow
    frame = getframe(fig);
    writeVideo(writer, frame);
end

% Close the writer
close(writer);


stx.String = " ";
exportgraphics(fig,FigurePath+FigureNameB)
