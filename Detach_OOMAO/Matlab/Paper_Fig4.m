addpath tools/functions
clear all;clc;close all

%% Preconditioners paths
DPWFSr1_path = "../Pytorch/training_results/Paper/06-07-2023/r1/DE/";
DPWFSr2_path = "../Pytorch/training_results/Paper/06-07-2023/r2/DE/";
DPWFSn1_path = "../Pytorch/training_results/Paper/06-07-2023/n1/DE/";

FigurePath = "./figures/paper/Figure4/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "Supplemental I.mp4";
FigureNameB = "Fig_4.pdf";

DE_list = dir(DPWFSr1_path+"*.mat");
DE_list = {DE_list.name};
DE_list = natsort(DE_list);

DE2_list = dir(DPWFSr2_path+"*.mat");
DE2_list = {DE2_list.name};
DE2_list = natsort(DE2_list);

DEn_list = dir(DPWFSn1_path+"*.mat");
DEn_list = {DEn_list.name};
DEn_list = natsort(DEn_list);

% stack
DE = ones(512,512,1);DEn = ones(512,512,1);DE2 = ones(512,512,1);
for k = 1:length(DE_list)-1
   load(DPWFSr1_path+DE_list{k})
   DE = cat(3,DE,fftshift(OL1)); 
   load(DPWFSn1_path+DEn_list{k})
   DEn = cat(3,DEn,fftshift(OL1));
end
for k = 1:length(DE_list)-1
   if k <= length(DE2_list)
   load(DPWFSr2_path+DE2_list{k})
   else
   load(DPWFSr2_path+DE2_list{length(DE2_list)})    
   end
   DE2 = cat(3,DE2,fftshift(OL1));  
   

end



%% Prepare plot

fig = figure('Color','w',Units='normalized',Position=[0.1589 0.3491 0.6719 0.4083]);

ha = tight_subplot(1,3,[0 .01],[.019 .001],[.02 .02]);

axes(ha(1))
im1 = imagesc(DE(:,:,2),[-pi pi]); colormap hsv;c1 = colorbar;axis image
axis off
tx1 = annotation('textbox',[0.134 0.75 0.3 0.15],"String","(a)",'FontSize',20);
tx1.LineStyle = 'none';tx1.FontWeight = 'bold';
c1.Visible = 'off';
ha(1).Position = [0.134 0.0190 0.2533 0.9800];


axes(ha(2))
im2 = imagesc(DE2(:,:,2),[-pi pi]); colormap hsv;c2 = colorbar;axis image
axis off
tx2 = annotation('textbox',[0.4 0.75 0.3 0.15],"String","(b)",'FontSize',20);
tx2.LineStyle = 'none';tx2.FontWeight = 'bold';
c2.Visible = 'off';
ha(2).Position = [0.4 0.0190 0.2533 0.9800];

axes(ha(3))
im3 = imagesc(DEn(:,:,2),[-pi pi]); colormap hsv;c3 = colorbar;axis image
axis off
tx3 = annotation('textbox',[0.67 0.75 0.3 0.15],"String","(c)",'FontSize',20);
tx3.LineStyle = 'none';tx3.FontWeight = 'bold';
c3.Ticks = [-pi 0 pi];
c3.TickLabels = {"-\pi","0","\pi"};
c3.FontSize = 24;

stx = sgtitle('Epoch 0','FontSize',22);

%%
writer = VideoWriter(FigurePath+FigureNameA, 'MPEG-4');
writer.FrameRate = 30;
frames = 500; % Number of frames
ff = 1:frames;
frame_idx = round((ff.^2.4)/(frames.^2.4)*99+1);

open(writer);
% Generate the animation
for fr = 1:frames
    im1.CData = DE(:,:,frame_idx(fr));
    im2.CData = DE2(:,:,frame_idx(fr));
    im3.CData = DEn(:,:,frame_idx(fr));
    stx.String = "Epoch "+frame_idx(fr);
    drawnow
    frame = getframe(fig);
    writeVideo(writer, frame);
end

% Close the writer
close(writer);


stx.String = " ";
exportgraphics(fig,FigurePath+FigureNameB)
