addpath tools/functions
clear all;clc

%% Preconditioners paths (Compute true to recalculate)
DPWFS_path = "../Preconditioners/nocap/base/checkpoint/OL1_R128_M0_RMSE0.0285_Epoch_92.mat";
DPWFSn_path = "../Preconditioners/nocap/pnoise/checkpoint/OL1_R128_M0_RMSE0.05275_Epoch_118.mat";
FigurePath = "./figures/Figure7/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureName = "ElementA.pdf";

nPxPup        = 128;               % number of pixels to describe the pupil

load(DPWFS_path);DPWFS_DE = OL1;
load(DPWFSn_path);DPWFSn_DE = OL1;

y1 = fftshift(DPWFS_DE);
y2 = fftshift(DPWFSn_DE);

%% Plot

inLims = [min([min(y1(:)) min(y2(:))]) max([max(y1(:)) max(y2(:))])];

fig = figure('Color','w','Position',[444 342 975 420]);
ha = tight_subplot(1,2,[.0 .0],[.005 .005],[0 .02]);

axes(ha(1));
im1 = imagesc(y1,inLims);colormap jet;axis image;axis off
c1 = colorbar();
c1.Visible ='off';
tx1 = annotation('textbox',[0.2 0.2 0.3 0.3],'String',"(a)",...
    'FitBoxToText','on','FontSize',20,'LineStyle','none');
axes(ha(2));
im2 = imagesc(y2,inLims);colormap jet;axis image;axis off
c2 = colorbar();
tx2 = annotation('textbox',[0.2 0.2 0.3 0.3],'String',"(b)",...
    'FitBoxToText','on','FontSize',20,'LineStyle','none');

tx1.Position = [0.0550 0.9000 0.0673 0.1038];
tx2.Position = [0.4900 0.9000 0.0673 0.1038];
ha(1).Position = [0.0550 0.0050 0.4287 0.9900];

c2.FontSize = 22;
c2.TickLabelInterpreter = 'latex';

exportgraphics(fig,FigurePath+FigureName)