%% Exploring discriminating image statistics: eyes vs stripes
% 
% 
% To do
% 
% redo the masks
% 
% separate contour stats
% 
% put stats measures in functions (options struct)
% 
% get the images from the generations
% 
% get the statistics on the images and compute averages
% 
% shuffle the genes and re-compute
% 
% plot against evolutions
% 
% 
%% Introduction
% 
% 
% When humans look at most images we are immediately struct by some essential 
% qualities. An image is brightly colored or dull, cluttered or uncluttered, it 
% has a highly regular patter nor an irregular pattern. It has texture or is mostly 
% smooth color fields. It is has strong edges or lacks them, or perhaps the edges 
% are all straight, or all curved. These are the kinds of things that jump out 
% at people and most of these aspects can be characterized quite easily. When 
% looking at the images with evolve from Alfa and Beto we get the sense that differences 
% between regions are subtle, IT prefers object-like things (bananas, faces etc.) 
% where as V1 prefers stripe-like patterns. Nonetheless we have prejudices from 
% what neuroscience believes it has discovered about the function of these regions. 
% We believe that the computational role of V1 neurons is to isolate small regions 
% of the visual field and provide information about the size, contrast, edge-orientation, 
% widths (frequencies), and colors of luminosity fields within the isolated region. 
% Higher visual areas are believed to interpret the collective firing of V1 neurons 
% as a kind of saliency map. V4 is somewhat similar to V1 but is also believed 
% to be tuned for simple geometric patterns. IT (inferior temporal cortex) is 
% the third area we record from. IT is crucial for object recognition and is believed  
% to be tuned to complex combinations of shape and color (e.g. faces, animals, 
% toys). It is believed that single cells in IT are indifferent to simple stimuli 
% like spots or drifting gratings. These beliefs lead us to investigate whether 
% images evolved for IT and V1 channels can be discriminated by simple visual 
% characteristics. However, the tools necessary to quantify simple visual characteristics 
% are also essential for analyzing more ineffable quantities. 
% 
% This document shows how to strip an image down to its basic parts which allows 
% us to characterize the simple details that jump out at us. Here we show the 
% use of unsupervised image segmentation (superpixels) to strip an image down 
% to basic color fields. We also show how to identify regions of high and low 
% texture, as well as quantifying the extent to which an image or image region 
% is textured. We also show how to extract only the dominant edges in a picture 
% and then to characterize each edge in terms of its orientation selectivity (straightness) 
% or in how "wiggly" it is (curvature). Having characterized the dominant edges 
% in a picture we can then score the image over all. Textures can be oriented 
% too, so in addition to characterizing the orientation selectivity of edges we 
% also use MATLAB's histogram of orientated gradients algorithm to capture a broader 
% view of the orientedness of images and produce an overall score. Lastly, any 
% 2D function (such as an image) can be decomposed into the superposition of sinusoidal 
% functions of many frequencies and in two orthogonal directions, the is the basis 
% of the FFT and DCT (discrete cosine transform). If an image is simple (e.g. 
% a drifting grating) it requires only a few sinusoidal functions. A Gabor patch 
% with a 45 degree orientation essentially requires two sinusoids whose frequencies 
% match the patch, plus several other sinusoids with much lower frequencies. The 
% two higher frequency sinusoids make 45 degree slanted bars and the lower frequency 
% sinusoids create an interference pattern that selects out a round "spot" for 
% the location of the patch. Thus a way to roughly measure of the complexity of 
% an image is to decompose it into sinusoids and see how many are required to 
% do a "good job" reconstructing the image. The number of required components 
% is a measure of "compressibility" of the image and simpler images are more compressible. 
% There is more that can be done with a decomposition of the image, an image with 
% strong horizontal features will require a lot of sinusoids that are horizontally 
% oriented. This can be pictured if you make an image where the rows and columns 
% correspond to frequencies and the pixel values correspond to the coefficients 
% of the sinusoids. Because a horizontally oriented sinusoid would seem to "ripple" 
% from top to bottom if allowed to oscillate in time the horizontal sinusoid frequencies 
% are enumerated on the vertical axis (the rows) and vice versa. Thus an original 
% image with strong horizontal features would become a decomposed image (an FFT 
% or DCT plot) with strong _vertical_ features, and an image with 45 degree features 
% makes an FFT with strong -45 degree features. Thus if you kept only the major 
% components in an FFT or DCT plot an image with high orientation selectivity 
% would be skewed along a particular direction. This feature, the shape of the 
% FFT or DCT is well known to scientists studying natural image statistics and 
% the "blobbiness" or "star-shapedness" of the FFT can roughly identify images 
% taken indoors, up-close, outdoors, etc. Thus the sinusoidal decomposition (or 
% more generally "wavelet decomposition") of images is a powerful tool in the 
% analysis of low-level image characteristics. 
% 
% The purpose of sharing this document is to familiarize the entire laboratory 
% with these basic methods and to share easily exportable functions that you can 
% immediately take to the analysis of your images. In particular visualizing image 
% montages as edges, textures, or segmented images, may help sharpen your focus 
% on the features that matter. However, the ultimate purpose of putting time and 
% energy into developing these methods is two fold: 1) give Carlos something he 
% can use immediately and 2) provide a foundational set of quantities we can incorporate 
% into manifold control. 
% 
% 
% 
% In terms of things that Carlos can use immediately, he may have already solved 
% the fundamental eyes vs stripes problem and several confounds may prevent low-level 
% image characteristics from solving that problem. These confounding factors involve 
% the details of the nature of receptive fields, possible biases in fc6 space, 
% and possible adversarial features that thwart some attempts to characterize 
% the shape of FFT or DCT plots. Additionally some of the ways of characterizing 
% texture or orientation are sensitive to the extent of visual clutter or contrast. 
% These factors will be highlighted as we proceed but the general consequence 
% is that we can never get a "clean signal". Each image seems to be a mixture 
% of aspects which are important to the neuron and aspects that are not important. 
% As a consequence the differences between cortical regions tend to be how much 
% variability exists among evolved images. This is visible in the last code block 
% which summarizes a handful of statistics using PCA, tSNE, and just plotting 
% them against each other. IT generally has no preference, but V1 often does and 
% V4 might or might not. There you can also see that statistics are also correlated. 
% By viewing images it seems that the algorithms capturing the statistics depend 
% some image properties themselves. In other words it is easier for HOG to identify 
% gradients in an uncluttered image than a cluttered one. 
% 
% There are three approaches to clarifying the relevant aspects: 1) demixing, 
% which is the use of independent component analysis to separate uncorrelated 
% image features, 2) higher-level analysis such as wavelet scattering, or other 
% feature extraction methods which are less useful for manifold control, 3) Using 
% saliency maps (or even receptive field (RF) maps) to discount features that 
% come from regions of the image with low-saliency or are at the extremities of 
% the RF. This works because many of the scores used already compute a weighted 
% average across image patches. These can simply be incorporated into the weighting 
% schemes. There are many possibilities here and it is time consuming to explore 
% them all. It is best to re-group and focus on a coherent research strategy, 
% then return and explore the possibilities most synergistic with the larger research 
% agenda. 
% 
% We also need a foundational set of quantities we can incorporate into manifold 
% control. Low-level features are essential to any control paradigms that seek 
% to avoid experimenter prejudices (e.g. only show Gabor patches then claiming 
% Gabor patches are important). Manifold control, as defined here, means that 
% if we are given an initial fc6 vector we can modify that vector to produce a 
% new vector and a new image which is incrementally different from the initial 
% fc6 vector but in a way that we can clearly compare against in other trial. 
% In other words, manifold control is the ability to hold image aspects constant 
% while we change only one thing at a time. The typical thinking is that either 
% a map of fc6 space needs to be constructed or a dimensionality reduction scheme 
% needs to be created that projects fc6 space onto some other more explicit space. 
% Either approach requires the ability to calculate a large repertoire of image 
% properties. Before considering how to pick image properties to focus on lets 
% consider a circumstance where it is totally unnecessary... What if every dimension 
% of fc6 space already had a name? There are 4096 dimensions, surely we can think 
% of 4096 image properties just by adding "iness" to the ends of words: purpliness, 
% rectilineariness, washedoutiness. If someone had already reliably associated 
% each dimension with an image property then we could simply tell CMAES not to 
% change some dimensions. Then we would be able to evolve along a controlled set 
% of images. Our goal with manifold control is essentially to label fc6 space. 
% 
% Unfortunately we also need to consider state dependence. Returning to our 
% imaginary prelabeled fc6 space, "state dependence" would mean that if an image 
% is already quite "purpley" then the "washedoutiness" dimension is more like 
% a "waviness" dimension. Differential equations can capture this, but only if 
% we are considering an optimization problem and a second order differential equation. 
% The second derivative captures how a gradient changes with respect to a change 
% in state. So if we are following a gradient to optimize "washedoutiness" which 
% is the 3rd component of fc6 space we would normally just increase that component. 
% The second derivative can tell us that if we increase the 3rd component too 
% far then the image starts to get "wavy" instead but that if we start increasing 
% the 4th component we can counter act that effect and continue to increase "washedoutiness". 
% In this example a gradient which previously pointed along the 3rd component 
% only would now point along the 4th component, and the second derivative is what 
% captures this information. This extended example of second derivatives sets 
% up an important point about manifold control and possessing a large repertoire 
% of features to calculate. 
% 
% If we want the ability to control evolution along meaningful dimensions we 
% need to have several objectives and explore fc6 space and evaluate the rate 
% of change between various points in fc6 space. If we have a small number of 
% image properties we will not capture the full expressiveness of fc6 space. If 
% we use high-level image features (like the male-to-female example) we can control 
% only according to a (prejudiced) mental model of what matters. We do not need 
% to explicitly model differential equations in fc6 space (though resnets and 
% ODEnets can do that), however any algorithmic approach we choose to mapping 
% fc6 space will require that we define "control parameters" and that's what a 
% large repertoire of low-level image statistics offers. Because image properties 
% are usually calculated on 2D image channels and these are color images and there 
% are alternative color spaces which capture distinct aspects (e.g. Hue vs RGB 
% vs Grayscale) any statistic we calculate is actually a minimum of 7 statistics 
% (3 for RGB space, 3 for HSV and 1 for grayscale). Therefore we can quickly find 
% ourselves dealing with a large number of dimensions and an expansive scope which 
% does not overly constrain fc6 space. 
% 
% It this document I provide 14 basic image statistics. However it lacks any 
% measure of clutter, image depth or openness and many other low-level features. 
% All measures are selected because they can be implemented without any external 
% toolboxes. I also provide several alternative representations of images that 
% I encourage everyone to use to get a different perspective on the evolutions. 
% I avoid for-loops where ever possible. This make the document more compact but 
% also makes the functions more exportable. They are implemented as nested anonymous 
% functions. I've provided a separate script containing the functions. Simply 
% include the script file as "*run importExplorationFunctions.m*" and you will 
% have the functions available to your script without having to maintain a library 
% of functions or script files and worrying about import paths. 
% 
% In summary, this document can support most labmember's need to find simplified 
% perspectives of images and rough summary statistics and it has helped to clarify 
% a strategy for manifold control. The strategy of manifold control which these 
% methods can support is to replace the fc6 space with another space consisting 
% of a large number of image statistics. The set of statistics needs to be large 
% enough, and coarse enough to allow the full expressivity of fc6 space. The individual 
% statistics need a useful interpretation. Once we have this space it is relatively 
% simple to translate between a vector specifying the values of each statistic 
% and an fc6 space vector. In some cases the statistic will be differentiable, 
% when it is not differentiable we can train a neural network to predict it from 
% images or fc6 vectors (the neural network would then be differentiable). Once 
% we have differentiable functions that can provide the values for these statistics 
% we can simply back propagate or use gradient descent to translate from a specified 
% set of statistics to a specific image. We can use that gradient descent or back-propagation 
% method to allow CMAES to operate in the labeled space while fc6 space generates 
% images. However, we do not have enough statistics. We need characteristics like 
% depth, clutter, contrast, multi-scale structure, statistics about the sizes 
% of objects, more diverse ways of identifying texture, various salience metrics, 
% and more. Once we have a large set of statistics we can then apply them to alternative 
% representations (e.g. superpixel segmentation) and then get alternative representations 
% for each channel of multiple colorspaces. Thus the number of statistics and 
% dimensions will multiply, while retaining interpretability. We can then train 
% images and use a process like "sparse PCA" or Shapely analysis to identify which 
% statistics matter most. Too some extent this already accomplished. I have shared 
% some plots which show that some pretty understandable statistics (image quality, 
% orientation selectivity, and hue complexity) show strong difference between 
% brain regions. When all three dimensions are plotted against each other it makes 
% a "bull's eye" with V1 showing higher selectivity that V4 or IT. All regions 
% are different according to variation of information and Kullbeck-Leibler divergence. 
% Since image quality roughly corresponds to clutter we can interpret this as 
% showing the V1 is more selective for levels of clutter, orientedness, and color 
% complexity than V4 or IT, while V4 is more selective than IT. This makes sense 
% since neural activity in IT responds to more complexity than V4 or V1 and V4 
% handles more visual complexity than V1. It's pretty clear, pretty strong and 
% with a bit more follow up ready to fill a niche in a paper if needed. 
% 
% 
%% 
% 
%% 
%% Essential functions
% 
% 
% In this section I use anonymous functions which we then use later. Were this 
% a python notebook I could just define normal functions. I could also provide 
% each function as a separate file, or a script which defines them. Matlab does 
% not let me define normal functions in a live script. By putting all the functions 
% in this document I can ensure that you can see and debug all the function, but 
% also that you are not burdened with a large folder that you need to add to your 
% path. At this stage I don't use anything that is not a built in matlab function 
% but in the future I will. 
% 
% A NOTE ABOUT SPEED: MATLAB has "distributed array" objects. These are used 
% for parallelization.  An example of creating one is to first initialize a parallel 
% environment (e.g.  *parpool(6)*, where 6 is the number of cores you want to 
% use) and then create a cell array (e.g. *StatsObj* below) and then cast it to 
% a distributed variable (e.g. *StatsObj=distributed(StatsObj)*). If you do that 
% then the analysis will run as much as 6 times faster if you assign 6 cores to 
% the task. Unfortunately the MATLAB on my work computer is not properly configure 
% and nothing can be run in parallel, so I wrote the lines of code necessary to 
% do this and commented it out. That is in the section where *StatsObj* is created. 
% Distributed objects are designed to be used in the *cellfun* or *arrayfun* functions 
% which then process the object in parallel instead of in serial. 
% 
% 
% 
% 
% 
% In this code block we create functions for the purpose of decomposing images 
% into modes and then either returning those modes or performing a summative analysis 
% on them (such as compressibility)

addpath(genpath(fullfile(pwd,'\ExternalSoftware\FEX_packages')))

% Define functions to use in the analysis
unwrap=@(x) x(:);
unwrapStack=@(imStack) reshape(imStack,[prod(size(imStack,[1:3])),size(imStack,4)]); % assumes a stack of images is 4D
rewrapStack=@(imStack,newsize) reshape(imStack,newsize);

% enable loopless functions by explointing "cellfun"
cellifyImStack=@(imStack) squeeze(mat2cell(imStack,size(imStack,1),size(imStack,2),size(imStack,3),ones(1,size(imStack,4))));
applyFunc2Stack=@(imStack,func4stack) cell2mat(permute(cellfun(@(x) func4stack(x),cellifyImStack(imStack),'UniformOutput',false),[2,3,4,1]));
applyFunc2Stack2=@(imStack,input2,func4stack) cell2mat(permute(cellfun(@(x) func4stack(x,input2),cellifyImStack(imStack),'UniformOutput',false),[2,3,4,1]));
apply2Chans=@(im,func2apply) cell2mat(cellfun(@(x) func2apply(im(:,:,x)),permute(num2cell(1:size(im,3)),[1,3,2]),'UniformOutput',false));

% change the image channels we analyze
uint8ify_helper=@(x) uint8(255*(x-min(x(:)))./range(x(:)));
uint8ify=@(x) uint8ify_helper(double(x)); % cast to a standard 256 integer color space
prepImag=@(x) mat2cell(cat(3,uint8ify(x),uint8ify(rgb2gray(x)),uint8ify(rgb2hsv(uint8ify(x)))),size(x,1),size(x,2),ones(1,7)); % get R,G,B,Grayscale (Gr),H,S,V image channels (RETURNS A CELL ARRAY FOR LOOPLESS PROCESSING)
prepImagMat=@(x) cat(3,x,rgb2gray(x),uint8(255*rgb2hsv(x))); % get R,G,B,Grayscale (Gr),H,S,V image channels (returns a 3D array instead of a cell with 7 levels in the 3rd dimension)


% simple statistics
logScaleCorr=@(x) -sign(x).*log(1-abs(x)); % convert to log scale
downTrend=@(x) corr(linspace(max(x),min(x),length(x))',x,'type','Spearman');
upTrend=@(x) corr(linspace(min(x),max(x),length(x))',x,'type','Spearman');
compRatio=@(x,frac) find((cumsum(sort(unwrap(x),'descend'))/sum(unwrap(x)))>=frac,1,'first')/numel(x);
compThresh=@(x,frac) max([-inf;x((cumsum(x)/sum(x))>=frac)]);
getPDF=@(x) histcounts(x(:),(-0.5:1:255.5),'Normalization','pdf');
getEntropy_helper=@(pdf) -sum(pdf.*log(pdf+min(pdf(pdf>0 & ~isnan(pdf))).*(pdf==0)/1e6));
getImEntropy=@(x) getEntropy_helper(getPDF(x));


% change image characteristics SINGLE IMAGES
pify=@(x) 2*pi*(x-min(x(:)))./range(x(:))-pi; % put in range -pi to pi
prcThrsh=@(x,thrsh)  min(x(:)).*(x<=prctile(x(:),thrsh))   +  x.*double(x>prctile(x(:),thrsh)); % threshold the image using a percentile
sigmoidify=@(x,center) tanh(2*pi*(x-center)./range(x(:))); % apply a simoid squashing function to pixel values
HSVify=@(x) rgb2hsv(x(:,:,[1,ceil(size(x,3)/2),size(x,3)])); % take a grayscale or RGB image and convert it to HSV space
hueify=@(x) hsv2rgb(cat(3,x(:,:,1),ones([size(x,[1,2]),2]))); % take an HSV image and make an RGB to display just the Hue channel
hueifyRGBoGr=@(x) hueify(HSVify(repmat(x,[1,1,3]))); % get an RGB approximation of the Hue channel of any RGB or grayscale image
hueSatChan_helper=@(x,chan) x(:,:,chan);
hueSatChan=@(x,chan) hueSatChan_helper(rgb2hsv(x),chan); % get just the desired channel of an HSV representation of a strictly RGB image
histEq=@(x) uint8ify(reshape(tiedrank(single(x(:))+single(rand(numel(x),1)*double(max(1,min(abs(diff(sort(x(:))))))/1e6))),size(x))); % equilize the histogram of an image or image channel
resizeIm=@(x) imresize(x,256/min(size(x,[1,2]))); % resize an image to 256 x 256
cropIm=@(x)  imcrop(x,[max(1,floor((size(x,2)-256)/2)),max(1,floor((size(x,1)-256)/2)),255,255]); % croup out a 256 x 256  portion from the center of an image
biniarizeOtsu=@(x) x>graythresh(x); % apply an Otsu biniarization
cast2NGrayLevels=@(x,N)  floor((x-min(x(:)))*0.99*ceil(N)/range(x(:)))*range(x(:))/ceil(N-1)+min(x(:)); % make an N level image from an image of any discretization level





% Change image characteristics STACK OF IMAGES
prepImagStack=@(imStack) applyFunc2Stack(imStack,prepImagMat); % get the 7 colorchannels to make a 256 x 256 x 7 x N stack of images (r,g,b,gr,h,s,v)
uint8ifyStack=@(imStack)   applyFunc2Stack(imStack,uint8ify);%
pifyStack=@(imStack)   applyFunc2Stack(imStack,pify);%
prcThreshStack=@(imStack,thrsh)  applyFunc2Stack2(imStack,thrsh,prcThrsh);%
sigmoidifyStack=@(imStack,center)     applyFunc2Stack2(imStack,center,sigmoidify);% sigmoidify a stack of images but include all the values in all the image channels
sigmoidifyStackSepCenters=@(imStack,ptile)     applyFunc2Stack2(imStack-prctile(imStack,ptile,[1,2]),0,sigmoidify);% sigmoidify a stack of image but treat each image channel separately
hueifyRGBoGrStack=@(imStack)   applyFunc2Stack(imStack,hueifyRGBoGr);%
hueifyStack=@(imStack)   applyFunc2Stack(imStack,hueify);%
hueSatChanStack=@(imStack,chan)     applyFunc2Stack2(imStack,chan,hueSatChan);%
histEqStack=@(imStack)     applyFunc2Stack(imStack,histEq);%
biniarizeOtsuStack=@(imStack) applyFunc2Stack(imStack,biniarizeOtsu);
cast2NGrayLevelsStack=@(imStack,N) applyFunc2Stack2(imStack,N,cast2NGrayLevels);%
rgb2grayStack=@(imStack) applyFunc2Stack(imStack,@(x) rgb2gray(x));
gaussFiltStack=@(imStack,gaussWindow) cell2mat(permute(cellfun(@(x) imgaussfilt(x,gaussWindow),cellifyImStack(imStack),'UniformOutput',false),[2,3,4,1]));
%% 
%% 
%% Receptive field 
% 
% 
% In this code block we define a single function *imFocusRF* that considers 
% a single image and a receptive field mask then snips out a square portion of 
% the image centered on the mask peak that contains 2/3 of the cumulative pixel 
% intensity of the _mask._ Essentially it focuses on the key part of the RF mask. 
% Whether it returns a large box or a small box depends on the extent of the RF 
% and whether the RF is completely contained in the image or if the image "missed". 
% This function, *imFocusRF*, works on _single_images, use *imFocusRFStack* to 
% process a 4D stack of images.


% this method of getting the receptive field uses a free aspect ratio
% this allows distortions that may evince receptive field size
RFmaskPtile=50; % focus the image on the RF mask points whose transimissibility is above this percentile (50=> keep the top 50% least opaque points of the mask )

BBox_helper_sizes=@(regProps) arrayfun(@(x) prod(x.BoundingBox(3:4)),regProps); % get the sizes of a list of bounding boxes in a regProp object
BBox_helper_maxNdx=@(regProps) find(BBox_helper_sizes(regProps)==max(BBox_helper_sizes(regProps))); % identify the largest bounding box of a regProp object
BBox_helper=@(regProps) regProps(BBox_helper_maxNdx(regProps)).BoundingBox; % get just the larges bounding box from a list of bounding boxes in a regProp object
imFocusRF_helperThresh=@(mask) prctile(mask(:),RFmaskPtile); % identify a threshold (used to limit the search for RF peaks)

% This method of getting the receptive field has a fixed aspect ratio and
% adjusts to capture 2/3 of the cumulative pixel values of the mask
% get the location of the mask peak of the largest area
peakLocator_helper_croppedSubs=@(croppedMask)  [find(max(max(croppedMask,[],2))==max(croppedMask,[],2),1,'first'),find(max(max(croppedMask,[],1))==max(croppedMask,[],1),1,'first')]; % get the subscripts of a mask after cropping
peakLocator_helper_unCroppedSubs=@(croppedMask,BBox) BBox(2:-1:1) + peakLocator_helper_croppedSubs(croppedMask); % get the subscrips of a cropped region with respect to the original uncropped image
peakLocator_helper_BoxFinder=@(mask)  max(1,floor(BBox_helper(regionprops(mask>imFocusRF_helperThresh(mask),'BoundingBox')))); % get a bounding box containing the bulk of the RF and likely RF peak
peakLocator_helper=@(mask,BBox) peakLocator_helper_unCroppedSubs(imcrop(mask,BBox),BBox);
peakLocator=@(mask) peakLocator_helper(mask,peakLocator_helper_BoxFinder(mask)); % get the peak of the largest continous RF mask region
% center a box of arbitrary width around arbitrary points (e.g. the peak)
boxPos_helper2=@(centerRC,width) [centerRC(1),centerRC(2)]+[-floor(width/2);floor(width/2)];
boxPos_helper=@(mask,pos) pos + (-pos(1,:)+1).*(pos(1,:)<1) + (size(mask,1)-pos(2,:)).*(pos(2,:)>size(mask,1));
boxPos=@(mask,centerRC,width) boxPos_helper(mask,boxPos_helper2(centerRC,width)); % get a position in subscripts of the box
boxCumulative=@(mask,pos) sum(unwrap(mask(pos(1,1):pos(2,1),pos(1,2):pos(2,2))))./sum(mask(:)); % get the fraction of cumulative pixel intensity contained in the box
% scale the box to capture 2/3 of the cumulative pixel values
boxPos2BBox=@(pos) [pos(1,2:-1:1),pos(2,2:-1:1)-pos(1,2:-1:1)+1]; % get the position specification format matlab's image processing tools prefer (USES XY instead of Rows and Columns)
optimumBBbox=@(mask,centerRC) boxPos2BBox(boxPos(mask,centerRC,...
    fminbnd(@(width) abs(   (2/3) - boxCumulative(mask,boxPos(mask,centerRC,width))  ),32,256))); % an optimization function to adjust the width of the box

% encapsulate all of the above
imFocusRF=@(x,bbox) imresize(imcrop(x,bbox),[256,256]);


% get the RF focused images
imFocusRFStack=@(imStack,bbox) uint8ifyStack(cell2mat(permute(cellfun(@(x) imFocusRF(x,bbox),cellifyImStack(imStack),'UniformOutput',false),[2,3,4,1])));

%% 
%% 
%% Edge detection, edge orientation selectivity and edge curvature
% 
% 
% This code block does two things. It gives you a good way to visualize the 
% dominant edges of an image and it returns two different statistics about the 
% edges. 
% 
% The focus on dominant edges is entirely because it is easier to eliminate 
% spurious edges (salt and pepper noise) and is easier to calculate statistics 
% with. The recommended use is encapsulated with *edgeDetStackVis* which performs 
% edge detection on a 4D stack of images (best to give it one image channel only). 
% It first blurs the image slightly to help remove more noise, then once it has 
% edges it thickens (AKA "dilate") them slightly because it looks better and smooths 
% them out. If you don't want this or want more control use *edgeDetStack* which 
% does no blurring or dilating and lets you pick the edge detection algorithm 
% (try "canny" instead of "sobel").
% 
% 
% 
% Next it defines edge curvature statistics. This is the most arduous set of 
% statistics to gather, simply because of the format MATLAB uses to specify edges. 
% The two statistics are:
%% 
% * Edge orientation selectivity
% * Edge curvature
%% 
% Both begin by performing skeletonization on the smoothed edges found with 
% *edgeDetStackVis*. This makes each edge only one pixel wide.
% 
% Orientation selectivity steps through each point on an edge and gathers the 
% orientation (-90 deg to +90 deg) of the tangent to the edge at each point. Next 
% it performs a histogram with 9 bins (the same as HOG). It calculates a selectivity 
% index for the histogram. (max(histogram)-min(histogram)) /( max(histogram)+ 
% min(histogram)). This is index is often used to calculate the selectivity of 
% neuronal firing rates to orientations. If any histogram bin is empty I remove 
% it from considerations as min(histogram). This is effectively the "peakiness" 
% of the histogram. 
% 
% Curvature is formally defined as the average magnitude of the second derivative 
% of a curve. The second derivative is the rate of change of the tangent to a 
% curve. The tangent is compactly described as the orientation of an infinitesimal 
% section of the curve. In practice the section of a curve (AKA edge) cannot be 
% too small or too large when dealing with images. If it is too small then small 
% edge detection errors will make big bumps in local curvature. If it is too large 
% then it may include intersections or neighbor edge segments in it's estimation. 
% Calculating curvature is similar to calculating orientation selectivity, except 
% for each point in the edge you calculate the difference between the tangent 
% orientation of that point and it's closest neighbors. I do not do formal finite 
% element derivative calculation, just the mean of the closest points. A point 
% with an orientation of +89 degrees and another point with orientation of -89 
% deg has a true difference of 2 deg but will show up as 178 deg. This annoying 
% MATLAB foible is corrected in my code. Once we have the local difference at 
% each point on an edge we take the natural log of the values (plus one). I found 
% by trial and error that the log of the local curvature produces better visual 
% agreement and better discrimination between images. Then I calculate the mean 
% of all their absolute values. 
% 
% 
% 
% Problems:
% 
% You will get many NaN or Infs. This is because the edges detected are too 
% small to be used in the calculation, or they are too close to the border but 
% not entirely in the border. If you perform edge detection improperly you will 
% errors about improperly indexing a variable. Just check that you have edges 
% in the image you feed the statistics gathering functions and that they are large 
% enough (see *minEdgeExpanse*) and their skeletonization is far enough away from 
% the image border.
% 
% 
%% Combining edge statistics into wholistic image statistics
% 
% 
% Each edge has its own curvature and selectivity values. How do we combine 
% them?
% 
% Looking at the color images you may seen regions with texture, and color fields 
% with obvious rectilinear boundaries. The edge detection algorithm may sprinkle 
% the texture region with short wiggly edges and may chop up the boundary of the 
% color fields. Worse it may try to trace a texture producing one extremely long 
% and extremely wiggly edge that meanders all over the image. So I compute a weighted 
% average using ad-hoc supplemental statistics. It may be better to use different 
% weighting. For example I can compute the saliency of image patches then assign 
% each edge an average saliency. However MATLAB has no built in saliency estimation 
% so I didn't implement that ... yet.
% 
% Supplemental statistics for edges. 
%% 
% * Smoothness/roughness
% * total length
%% 
% Long edges are probably important edges. Rough edges that look like coastlines 
% are probably spurious edges. 
% 
% I assign each edge a smoothness score using an approach inspired by box-counting 
% dimension. The actual box-counting dimension of edges is probably a lot more 
% valuable, but there is no built in MATLAB implementation. Instead I find a convex 
% hull containing the edge and compute ratio of the number of pixels in the convex 
% hull to the number of pixels in the edge. Rectilinear edges and graceful curves 
% will have a large ratio. Crumpled string will have lower (but still kind of 
% large) ratios. Once I have  a vector specifying the smoothness of all edges 
% square the values then apply pdf normalization to that vector (must sum to one). 
% 
% I also calculate the total number of pixels in an edge. Since the edge is 
% one pixel wide this is the total length of the edge. Once I have a vector specifying 
% the length of edges I take natural logarithm and apply pdf normalization to 
% that vector (must sum to one). 
% 
% If the edge detection algorithm traced the texture in a part of the image 
% it would produce and extremely long but also very rough. By taking the log of 
% edge length I de-emphasize the longest edges and by taking the square of smoothness 
% I exaggerate the importance of smoothness in edges. Hopefully this balances. 
% I compute the geometric mean (sqrt(x*y)) of the length and smoothness vectors 
% and then apply pdf normalization to the resulting vector. This is then my weighting 
% vector. I multiply it by the selectivity and curvature values and then compute 
% the weighted average. This should give a statistic that emphasizes long graceful 
% edges and rejects long and short rough edges. 
% 
% 
% 
% 
% 
% *edgeDetStackVis* convert a 4D stack of images into a format suitable for 
% calculating overall image edge statistics (give it grayscale or single color 
% channel images)
% 
% *getLongEdges* returns only the dominant edges in a single image from the 
% stack returned by *edgeDetStackVis*
% 
% *getSimplifiedEdge* skeletonize one edge from the set of edges returned by 
% *getLongEdges* and return a logical (BW) image
% 
% *getPathSmoothness* compute the smoothness of the edge image returned by *getSimplifiedEdge*
% 
% *getPathLength* compute the smoothness of the edge image returned by *getSimplifiedEdge*
% 
% *getPathOrientations* get the local orientations of all the valid points (not 
% in image border) for the edge image returned by *getSimplifiedEdge*
% 
% *getPathCurvature* get the local curvature (w/out log transforming) of all 
% the valid points (not in image border) for the edge image returned by *getSimplifiedEdge*
% 
% *edgeOrientVis* get the linear indices of all the points in all the edges 
% of a single image from the stack returned by *edgeDetStackVis* AND get the local 
% orientation for all the points (an example of creating a figure from this is 
% below)
% 
% *edgeCurveVis* get the linear indices of all the points in all the edges of 
% a single image from the stack returned by *edgeDetStackVis* AND get the local 
% curvature for all the points (an example of creating a figure from this is below)
% 
% *getPathStats* get a column vector containing the selectivity, curvature, 
% smoothness and edge length for the edge image returned by *getSimplifiedEdge* 
% (everything appropriately transformed as described above)
% 
% *getEdgeStats* the same as *getPathStats* except for all the dominant edges 
% in a single image from the stack returned by *edgeDetStackVis*
% 
% *getCurvSelIndices* compute the weighted average to get the overall selectivity 
% and curvature in a single image from the stack returned by *edgeDetStackVisgetCurvSelIndicesStack*the 
% same as *getCurvSelIndices* except that it computes the selectivity and curvature 
% indexes for the entire stack returned by *edgeDetStackVis*
% 
% 

% perform edge detection, recommendend to use "sobel" which only captures
% STRONG edges. "canny" captures too many edges to analyze effectively.

% edge detection
getEdges=@(im,edgMeth) uint8ify(edge(im,edgMeth));
% each color channel separately
edgeDetSepChan=@(im,edgMeth) apply2Chans(im, @(im2) getEdges(im2,edgMeth) );
edgeDetStack=@(imStack,edgMeth) applyFunc2Stack2(imStack,edgMeth,edgeDetSepChan);
% recommended to use imgaussfilt(imStack,1.5) and the 'Sobel' method
% followed by imdilate(newStack,strel('disk',1))
edgeDetStackVis_helper=@(imStack) applyFunc2Stack(imStack,@(im) imdilate(im,strel('disk',1)));
edgeDetStackVis=@(imStack) edgeDetStackVis_helper(edgeDetStack(imgaussfilt(imStack,1.5),'Sobel'));




% Capture statistics about the detected edges

% define some fixed parameters
minEdgeExpanse=32; % a box containing the edge must have an edge longer than minEdgeExpanse to be included in the analysis
orientationWidth=9; % the size of the region used to estimate the tangent to the edge
excWindow=floor(orientationWidth/2); % we cannot calculate the tangent for points closer to the boundary of the image than this (MUST be set >= floor(orientationWidth/2))

% eliminate short edges or edges entirely contained in the image border
getSubs=@(x,im) [mod(x-1,size(im,1))+1,ceil(x/size(im,2))];
removeEdgesInBorder=@(subs,im,orientationWidth) subs( all(subs>ceil(orientationWidth/2) & subs< (repmat(size(im),size(subs,1),1)-ceil(orientationWidth/2)),2) , : );
% getLineArea_helper=@(x) prod(max(max(max(x,[],1)-min(x,[],1))^2,0)); % the odd expression here is a le
getLineArea_helper=@(x) prod(max([max(max(x,[],1)-min(x,[],1)).^2,0])); % the odd expression here is a le
getLineAreas=@(pixList,im,orientationWidth) cellfun(@(x) getLineArea_helper(removeEdgesInBorder(getSubs(x,im),im,orientationWidth)), pixList);
getPixList_helper=@(x) x.PixelIdxList;
getPixList=@(im) getPixList_helper(bwconncomp(im));
getLongEdges_helper=@(im,pixList,excWindow,orientationWidth) pixList(getLineAreas(pixList,im,orientationWidth)>minEdgeExpanse^2);
getLongEdges=@(im, excWindow,orientationWidth) getLongEdges_helper(im,getPixList(im), excWindow,orientationWidth);


getSimplifiedEdge=@(im,pixList) bwskel(ismember(reshape(1:numel(im),size(im)),pixList));
getPathSmoothness=@(edgeIm) (sum(bwconvhull(edgeIm),'all')./sum(edgeIm(:)))^2;
getPathLength=@(edgeIm) sum(edgeIm,'all');

getPathPoints2Test=@(edgeIm,borderExlusion) find(logical(edgeIm.*(padarray(ones(size(edgeIm)-2*borderExlusion),borderExlusion*ones(1,2),0))));

getPointOrientation_helper=@(edgeIm,subs,orientationWidth)  regionprops(edgeIm(subs(1)+[-floor(orientationWidth/2):(ceil(orientationWidth/2)-1)],subs(2)+[-floor(orientationWidth/2):(ceil(orientationWidth/2)-1)]),'Orientation');
getPointOrientation_helper2=@(boxStats) boxStats.Orientation;
getPointOrientation=@(edgeIm,subs,orientationWidth)  getPointOrientation_helper2(getPointOrientation_helper(edgeIm,subs,orientationWidth));
getPathOrientations=@(edgeIm,subs,orientationWidth) cellfun(@(x) getPointOrientation(edgeIm,x,orientationWidth),mat2cell(subs,ones(size(subs,1),1),2));

getPathNeighbors_helperFixEmpty2=@(neighbs,K) (numel(neighbs)<2) & true(1,K-numel(neighbs));
getPathNeighbors_helperFixEmpty=@(neighbs,K,replaceLabels)  [neighbs,replaceLabels(getPathNeighbors_helperFixEmpty2(neighbs,K))];
getPathNeighbors_helper=@(neighborNdcs,K)  cellfun(@(x) getPathNeighbors_helperFixEmpty(x(:,1:end),K,ones(1,K-numel(x))) , neighborNdcs,'UniformOutput',false);
getPathNeighbors=@(subs,K) getPathNeighbors_helper(knnsearch(subs,subs,'K',K,'Distance','euclidean','IncludeTies',true),K);

getPointOrientationDiff_helper=@(neighbOrientations) [neighbOrientations(2:end)';neighbOrientations(2:end)'+(neighbOrientations(1)==0 || sign(neighbOrientations(1)))*180]-neighbOrientations(1);
getPointOrientationDiff_helper2=@(diffs) diffs(abs(diffs)==min(abs(diffs),[],1) & [true(1,size(diffs,2));diff(abs(diffs),[],1)~=0]  )';
getPointNeighborDistances=@(subs,neighbors) sqrt(sum((subs(neighbors(1),:)-subs(neighbors(2:end),:)).^2,2))';
getPointOrientationDiff=@(subs,neighbors,neighbOrientations) getPointOrientationDiff_helper2(getPointOrientationDiff_helper(neighbOrientations))./getPointNeighborDistances(subs,neighbors);

getPointOrientationDeriv=@(subs,neighbors,neighbOrientations) nanmean(getPointOrientationDiff(subs,neighbors,neighbOrientations));
getPathCurvature=@(subs,allNeighbors,orientations) cellfun(@(x)  abs(getPointOrientationDeriv(subs,x,orientations(x))), allNeighbors  );


getPathOrientationSelectivity_helper=@(orientDist) (max(orientDist)-min(orientDist(orientDist~=0)))./(max(orientDist)+min(orientDist(orientDist~=0)));
getPathOrientationSelectivity=@(orientations) getPathOrientationSelectivity_helper(histcounts(orientations,linspace(-90,90,10),'Normalization','pdf'));


edgeOrientVis_helper2=@(edgeIm,ndcs,orientationWidth) [ndcs,getPathOrientations(edgeIm,getSubs(ndcs,edgeIm),orientationWidth)];
edgeOrientVis_helper=@(edgeIm,orientationWidth) edgeOrientVis_helper2(edgeIm,getPathPoints2Test(edgeIm,floor(orientationWidth/2)),orientationWidth);
edgeOrientVis=@(bwIm,excWindow,orientationWidth) cell2mat(cellfun(@(x) edgeOrientVis_helper(getSimplifiedEdge(bwIm,x),orientationWidth),getLongEdges(bwIm,excWindow,orientationWidth)','UniformOutput',false  ));



edgeCurveVis_helper2=@(edgeIm,ndcs,orientationWidth) [ndcs,getPathCurvature(getSubs(ndcs,edgeIm),getPathNeighbors(getSubs(ndcs,edgeIm),orientationWidth),getPathOrientations(edgeIm,getSubs(ndcs,edgeIm),orientationWidth))];
edgeCurveVis_helper=@(edgeIm,orientationWidth) edgeCurveVis_helper2(edgeIm,getPathPoints2Test(edgeIm,floor(orientationWidth/2)),orientationWidth);
edgeCurveVis=@(bwIm,excWindow,orientationWidth) cell2mat(cellfun(@(x) edgeCurveVis_helper(getSimplifiedEdge(bwIm,x),orientationWidth),getLongEdges(bwIm,excWindow,orientationWidth)','UniformOutput',false  ));



getPathStats_helper2=@(edgeIm,subs,allNeighbors,orientations)  [max([-inf,getPathOrientationSelectivity(orientations)]);max([-inf,mean(log(1+getPathCurvature(subs,allNeighbors,orientations)))]);max([-inf,getPathSmoothness(edgeIm)]);max([-inf,getPathLength(edgeIm)])];
getPathStats_helper3=@(edgeIm,subs,orientationWidth) getPathStats_helper2(edgeIm,subs,getPathNeighbors(subs,orientationWidth),getPathOrientations(edgeIm,subs,orientationWidth));
getPathStats=@(edgeIm,orientationWidth) getPathStats_helper3(edgeIm,getSubs(getPathPoints2Test(edgeIm,floor(orientationWidth/2)),edgeIm),orientationWidth);

getEdgeStats_helper=@(edgeStats) [edgeStats,nan(4*isempty(edgeStats),1)];
getEdgeStats=@(bwIm,excWindow,orientationWidth) getEdgeStats_helper(cell2mat(cellfun(@(x) getPathStats(getSimplifiedEdge(bwIm,x),orientationWidth),getLongEdges(bwIm,excWindow,orientationWidth),'UniformOutput',false  )));

getCurvSelIndices_helperGetWeighting2=@(unNormWeighting) unNormWeighting./sum(unNormWeighting);
getCurvSelIndices_helperGetWeighting=@(edgeStats) getCurvSelIndices_helperGetWeighting2(sqrt((edgeStats(3,:)./sum(edgeStats(3,:))).*(edgeStats(4,:)./sum(edgeStats(4,:)))));
getCurvSelIndices_helper2=@(indices)  indices.*~isinf(indices);
getCurvSelIndices_helper=@(edgeStats) getCurvSelIndices_helper2(sum(edgeStats(1:2,:).*repmat(getCurvSelIndices_helperGetWeighting(edgeStats),2,1),2));

getCurvSelIndices=@(bwIm,excWindow,orientationWidth) getCurvSelIndices_helper(getEdgeStats(bwIm,excWindow,orientationWidth));

getCurvSelIndicesStack=@(imStack,excWindow,orientationWidth) applyFunc2Stack2(imStack,{excWindow,orientationWidth},@(im,x) getCurvSelIndices(im,x{1},x{2}));

%% 
%% 
%% Unsupervised image segmentation
% 
% 
% This effectively "coarse grains" an image, eliminating details like texture. 
% If your hypothesis is that a neuron is looking at the stripy color fields in 
% the center of an image and ignoring the cluttered nonsense around the edges 
% then this "superpixel" representation will return something closer to what you 
% think the neuron cares about. 
% 
% Superpixels are groups of pixels. They are found through a process that is 
% essentially k-means on RGB pixels. When used on grayscale images it works similarly. 
% The key difference is that the XY distance between pixels is accounted for when 
% deciding whether two pixel are in the same group or not. If you just looked 
% at pixel values you would be combining pixels scattered across the whole image. 
% This latter part is actually also a valid approach, but it is used to reduce 
% the color levels of an image.
% 
% Because you can perform super pixel segmentation on either the full RGB space 
% or on just a single color channel there are two implementations. 
% 
% 
% 
% *K* => desired (not guaranteed) number of superpixel in the final segmentation 
% (an input argument for all the following functions)
% 
% *supPixSegCombChan* returns a segmented image based on a segmentation using 
% the combined RGB space (must be RGB) (visualization is very coherent)
% 
% *supPixSegSepChan* returns a segmented image based on a segmentation performed 
% on each of the RGB channels separately. (emphasizes differences between color 
% channels, less coherent final visualization)*supPixSegStack* performs *supPixSegSepChan*on 
% a 4D stack of images. It does not care about the color space of the stack because 
% it works on channels separately. 


% unsupervised segmentation (super pixels)
getSupPix=@(im,k) superpixels(im,k);
supPixSeg_helper=@(im,supPix)  sum(cell2mat(cellfun(@(x)  (supPix==x).*mean(unwrap(im(supPix==x))), permute(num2cell(unique(supPix)'),[1,3,4,2]),'UniformOutput',false)),4);
% combined color channels
supPixSegCombChan=@(im,k) supPixSeg_helper(im, getSupPix(im,k));
% each color channel separately
supPixSegSepChan=@(im,k) apply2Chans(im, @(im2) supPixSeg_helper(im2,getSupPix(im2,k)) );
supPixSegStack=@(imStack,k) uint8ifyStack(applyFunc2Stack2(imStack,k,supPixSegSepChan));

%% 
%% 
%% Image texture index
% 
% 
% One of the key tools in identifying texture regions in an image is simply 
% to compute the entropy of image patches. Higher entropy implies higher variability 
% and thus the region has more texture. However humans regard bi-color stripes 
% as highly textured. Entropy regards stripes as very predictable. In practice 
% entropy works well because stripes are never very uniform, but in theory this 
% could classify stripy patterns as non-textured. 
% 
% This calculation performs superpixel segmentation then calculates the entropy 
% of each superpixel. This produced a new superpixel representation where regions 
% of high texture are emphasized with white and low texture (i.e. color fields) 
% are emphasized in black. An overall texture index is computed using a weighted 
% average of the entropies across superpixels, the weighting is provided by the 
% area of the superpixels. It may be better to use saliency as an alternative 
% weighting, or two combine spectral information to control for bi-color stripes. 
% This is not implemented yet.
% 
% Again this can be performed on individual image channels or all at once. 
% 
% 
% 
% *texIndexVisCombChan*  Create a 2D image like mapping of the entropies of 
% each superpixel resulting segmentation on all color channels (the units are 
% in bits, use *uint8ify* before using *imshow*, consider using *sigmoidify* for 
% visual clarity)    
% 
% *texIndexCombChan*  Calculate the texture index for the entirety of an RGB 
% image     
% 
% *texIndexVisSepChan*Create a 2D image like mapping of the entropies of each 
% superpixel resulting segmentation on individual color channels, and thus individual 
% entropy maps (the units are in bits, use *uint8ify*before using *imshow*, consider 
% using *sigmoidify* for visual clarity)    
% 
% *texIndexSepChan*   Calculate the texture index for each image channel of 
% an image  (any uint8 color space)
% 
% *texIndexVisStack* perform *texIndexVisSepChan* on a 4D stack of images and  
% apply *sigmoidify* to each channel and image (the units are in bits, use *uint8ify*before 
% using *imshow*)
% 
% *texIndexStack*     perform *texIndexSepChan* on a 4D stack of images (returns 
% index values, not a 2D map)*texIndexCombChanStack*   perform *texIndexCombChan*on 
% a 4D stack of images (returns index values, not a 2D map)


% get a texturization index (the weighted average of entropy within each
% superpixel)
texIndexVis_helper=@(im,supPix) sum(cell2mat(cellfun(@(x)  (supPix==x).*getImEntropy(unwrap(double(im(supPix==x)))), permute(num2cell(unique(supPix)'),[1,3,4,2]),'UniformOutput',false)),4);
texIndex_helper=@(im,supPix) sum(cell2mat(cellfun(@(x)  sum(unwrap(supPix==x)).*getImEntropy(unwrap(double(im(repmat(supPix==x,[1,1,size(im,3)]))))), permute(num2cell(unique(supPix)'),[1,3,4,2]),'UniformOutput',false)),4)./numel(im);
% the weighted average of superpixel entropy
texIndexVisCombChan=@(im,k) texIndexVis_helper(im, getSupPix(im,k));
texIndexCombChan=@(im,k) texIndex_helper(im, getSupPix(im,k));
% each color channel separately
texIndexVisSepChan=@(im,k) apply2Chans(im, @(im2) texIndexVis_helper(im2,getSupPix(im2,k)) );
texIndexSepChan=@(im,k) apply2Chans(im, @(im2) texIndex_helper(im2,getSupPix(im2,k)) );
texIndexVisStack=@(imStack,k) sigmoidifyStackSepCenters(applyFunc2Stack2(imStack,k,texIndexVisSepChan),75);
texIndexStack=@(imStack,k) applyFunc2Stack2(imStack,k,texIndexSepChan);
texIndexCombChanStack=@(imStack,k) applyFunc2Stack2(imStack,k,texIndexCombChan);
%% 
%% 
%% Entropy filters and analysis
% 
% 
% This is probably one of the most valuable visualization for understand the 
% evolutionary strategies found by CMAES. 
% 
% Entropy is a measure of variability that is agnostic to pixel values. A binary 
% image flip-flopping between 0 and 255 will have the same entropy if you rescale 
% it to instead flip-flop between 64 and 192. If you the pixel standard deviation 
% or other classical measures of dispersion you would get very different answers.  
% However, for stability I created by own implementation of entropy that assumes 
% a uint8 0-255 color space. MATLAB's implementation will adapt its probability 
% space to what ever set of values it's given and that was found to make comparisons 
% across image categories difficult. Over all entropy (*getEntropy*) is useful 
% to perform on specific color channels and revealed great diversity among the 
% patterns of Hue in images, as well as clear differentiation among the R G B 
% channels. 
% 
% An entropy filter (*entFilt*, and *entFiltStack*) calculates the entropy in 
% local patches centered on each extant pixel and then replaces that pixel with 
% calculated entropy value. It is often helpful to use sigmoid squashing to emphasize 
% high-entropy pixels. This better exposes edges and region boundaries. The center 
% of the sigmoid (the break point) is specified with a percentile as *entPtileSigmoidCenter*. 
% The size of the patch is controlled with *entWindow*
% 
% An entropy trend (*entTrend*) looks across a stack of images. For each pixel 
% it calculates the entropy of all the pixels in that same location across part 
% of a stack of images. You can specify the width of that part using *entWindow*. 
% By looking at the entropy across images in a stack we can see whether one pixel 
% or another is frequently modified in the different versions. If a region is 
% particularly important it will be unchanged among all the "most fit" images 
% while unimportant pixels will be allowed to change more often. Because *entTrend* 
% is computationally expensive I give you the option of doing fewer calculations. 
% If you set *entWindow*=20, and *entStep*=5 it will look at the first 20 images 
% in the stack (images 1-20) and compute the entropy trend of each pixel, then 
% rather than looking at the next 20 images (images 2-21) it will skip 5 images 
% (images 5-24). 
% 
% I provide movies where you can watch entropy change across generations. By 
% comparing entropy filters to entropy trends you can get a sense of what the 
% evolutionary strategy is actually doing. It was also useful for retrodicting 
% receptive fields, where I found that early on entropy was higher in the peak 
% of the RF as the algorithm quickly settled to a solution. Then often, though 
% not always, entropy dropped precipitously in the peak RF region. You can also 
% watch regions expand and contract very clearly by using either entropy trend 
% or entropy filter. 
% 
% 
% 
% *imChanEntropy*     get the entropy of a single image in each of the  R G 
% B Gr H S V color channels
% 
% *entropyByGenYimChan*     apply *imChanEntropy*to a 4D stack of RGB images
% 
% *entropyStack*    get the entropy of a stack of _single channel_ images (if 
% you feed it multi channel images it will compute entropy by pooling the channels)
% 
% *hueEntropy*     get the entropy of just the hue channel of a single RGB image
% 
% *hueEntropyStack*     get the entropy of just the hue channel of a stack of 
% RGB images



% entropy filter

entPtileSigmoidCenter=75; % after entropy filter values above this percentile will be white and those below will be black because we apply a sigmoid (tanh+1) to pixel values
entFilt=@(im,entWindow) uint8ify(entropyfilt(im,true(entWindow)));
entFiltStack=@(imStack,entWindow) applyFunc2Stack2(imStack,entWindow,entFilt);

% entropy of evolution (or trends across stacks)
entTrend_helperWindow=@(imStackUnwrapped) cell2mat(cellfun(@(x) getImEntropy(imStackUnwrapped(x,:)),num2cell((1:size(imStackUnwrapped,1))'),'UniformOutput',false));
% entTrend_helperWindow=@(imStackUnwrapped) cell2mat(cellfun(@(x) entropy(double(imStackUnwrapped(x,:))),num2cell((1:size(imStackUnwrapped,1))'),'UniformOutput',false))
entTrend=@(imStack,entWindow,entStep) cell2mat(cellfun(@(x) uint8ifyStack(sigmoidifyStackSepCenters(  rewrapStack(  entTrend_helperWindow(unwrapStack(imStack(:,:,:,x+(1:entWindow)-1)))  ,  [size(imStack,[1:3]),1]  )  ,entPtileSigmoidCenter)), permute(num2cell(1:entStep:(1+size(imStack,4)-entWindow)),[1,3,4,2]),'UniformOutput',false));


% get entropy
imChanEntropy=@(x) squeeze(cellfun(@(x) getImEntropy(double(x)),prepImag(x)))';
entropyByGenYimChan=@(imStack) cell2mat(cellfun(@(x) imChanEntropy(x),cellifyImStack(imStack),'UniformOutput',false));
% entropyStack=@(imStack) applyFunc2Stack(imStack,@(x) entropy(x));
entropyStack=@(imStack) applyFunc2Stack(imStack,@(x) getImEntropy(x));
HueChan=@(x) x{5};
hueEntropy=@(im) entropy(HueChan(prepImag(im)));
hueEntropyStack=@(imStack) applyFunc2Stack(imStack,hueEntropy);

%% 
%% 
%% HOG: Histogram of oriented gradients
% 
% 
% Recall that the FFT shows signatures of specific orientations, well the HOG 
% algorithm decomposes small patches (controlled by *cellSize*) of an image into 
% orthogonal components (called gradients) which emphasizes pixel intensity at 
% the edges of the block. Then it identifies 9 orientations from 0 to 180 and 
% estimates the probability that the pixels in that patch represent an edge at 
% that orientation. To accommodate changes in illumination and contrast across 
% the image the patches are supposed to be grouped. However MATLAB does a poor 
% job of separating the results of the groupings from the results of individual 
% cells. So in my implementation I ignore groupings. For each patch the resulting 
% probability distribution can be used to calculate the orientation selectivity 
% each patch. Thus we can compute an over all orientation selectivity. If we had 
% the data from the patch groupings we could also calculate a confidence value 
% for each patch and use that as a weighting scheme. In lieu of a weighting scheme 
% I simply compute the mean selectivity of the top N patches as identified with 
% a percentile (*HOGptile,* 75=> means selectivity of the top 25% most selective 
% patches). This function is *meanHOGSel* and *meanHOGSelStack*. An alternative 
% statistics is to combine the HOG patches in such a way as to compute the overall 
% selectivity this is implemented with *totalHOGSel* and *totalHOGSelStack*. Visualizing 
% the HOG representation is easy, visualizing a few key HOG patches is not straightforward 
% (again I'm holding MathWorks responsible). This is demonstrated in the visualizations 
% below.  
% 
% 

cellSize=16;
HOGptile=75;

% get the HOG selectivity statistics
getHOGStats_helper=@(im,cellSize) reshape(extractHOGFeatures(im,'CellSize',[cellSize,cellSize],'BlockSize',[1,1]),9,prod(floor(size(im,[1,2])/cellSize)));
getHOGStats_helperNoZeroMin=@(HOG) HOG+100*max(HOG(:)).*(HOG==0);
getHOGStats_helperSel=@(HOG) (max(HOG,[],1)-min(getHOGStats_helperNoZeroMin(HOG),[],1))./(max(HOG,[],1)+min(getHOGStats_helperNoZeroMin(HOG),[],1));
% get the mean selectivity of the most selective HOGs
meanHOGSel_helper=@(selVec,ptile) mean(selVec(selVec>=prctile(selVec,ptile)));
meanHOGSel=@(im,cellSize,ptile) meanHOGSel_helper(getHOGStats_helperSel(getHOGStats_helper(im,cellSize)),ptile);
% combine the most selective HOGs and get an overall selectivity
totalHOGSel_helper=@(HOG,selVec,ptile) sum(HOG(:,selVec>=prctile(selVec,ptile)),2)./sum(HOG(:,selVec>=prctile(selVec,ptile)),'all');
totalHOGSel_helper1=@(HOG,ptile) totalHOGSel_helper(HOG,getHOGStats_helperSel(HOG),ptile);
totalHOGSel=@(im,cellSize,ptile)  getHOGStats_helperSel(totalHOGSel_helper1(getHOGStats_helper(im,cellSize),ptile));
% get HOG stats of a stack of images
meanHOGSelStack=@(imStack,cellSize,ptile) applyFunc2Stack2(imStack,{cellSize,ptile},@(im,x) meanHOGSel(im,x{1},x{2}));
totalHOGSelStack=@(imStack,cellSize,ptile) applyFunc2Stack2(imStack,{cellSize,ptile},@(im,x) totalHOGSel(im,x{1},x{2}));
%% 
%% 
%% Perceptual image quality
% 
% 
% This the only high-level statistic here. I selected it because it is built 
% in to MATLAB and based on many of the same principles used. Other image quality 
% metrics may allow discrimination but they are not constructed from pure maths. 
% Instead the researchers used a statistical model that depends on a either a 
% specific reference image or a repository of images. A low score is a good image 
% and a high score is a bad one. Some of the evolved images are actually comparable 
% to typical photographs (PIQE~20) these tend to be uncluttered and have a white 
% space or color field on top of a textured background.  PIQE and HueEntropy are 
% highly anticorrelated, indicating that high HueEntropy is associated with clutter 
% and murky images. 
% 
% 

% perceptual image quality metrics

piqeStack=@(imStack) applyFunc2Stack(imStack,@(x) piqe(x));
%% 
%% 
%% Frequency decomposition
% 
% 
% Here I demonstrate how to visualize and roughly quantify the DCT and FFT transform 
% of images. Because the alternative image representations (edge detection, superpixel 
% segmentations, and entropy filters) often remove aspects of an image which may 
% skew the spectral representation it makes sense to try to look at their spectra, 
% as well as the spectra of original images. Additionally both the DCT and FFT 
% are sensitive to the distribution of pixel values, preprocessing like histogram 
% equalization *histEq*, or application to different color channels or spaces 
% can produce very different results. However there are so many permutations to 
% try and it is so time consuming that they were not extensively tested. I encourage 
% you to try. 
% 
% 
% 
% To accommodate trying different representations and transformations I do two 
% things: 1) simplify representation and transformation into a simple functional 
% composition. 2) capture the functions which depend on the transformation and 
% representation and provide a one-line command to update them all with the new 
% selection. 
% 
% The functional composition is articulated in the first lines below and the 
% ability to recreate the functions and update the transformations is simple: 
% run the next TWO code blocks below (this one and "spectral statistics") and 
% then when you redefine a functional composition ALWAYS follow it with the command 
% *eval(recreateFunctionsString)*.
% 
% 
% 
% 
% 
% The main difference between FFT and DCT is that the FFT is complex valued 
% but we can only simultaneously plot (and feed our algorithms with) real values 
% so you miss the information contained in the imaginary part. The DCT (discrete 
% cosine transform) is complete, containing all the same information as the FFT 
% but it is intrinsically real valued. The DCT places the zero-zero frequency 
% origin in the upper left (or row 1 column 1) but the FFT places in in the center 
% of the plot (for us this is row 128 column 128). Consequently the DCT plots 
% visual features aligned with 45 degrees and 135 degrees along the same ray extending 
% along -45 degrees, where as FFT would plot them along the 135 degree and 45 
% degree rays respectively. Thus the overall shape of the DCT is somewhat more 
% crude and less able to differentiate features based solely on the shape of the 
% plot. Typically both are used as the starting point for a machine learning algorithm 
% such as SVD or simple perceptron networks. 
% 
% The most basic way to quantify an FFT or DCT transformation is to see how 
% "concentrated" it is. Each is a dimensionality reduction method for images. 
% So you could see how many components are required to capture X% of the energy 
% in the DCT or FFT. This is exactly the same of seeing how many PCA components 
% contain X% of the energy in the eigen decomposition. Another common approach 
% is to keep the largest X% of components and compute the ratio of the energy 
% in those components to the rest of the components. The last method is to capture 
% all components within X% of the largest component (it is especially important 
% to subtract the mean of the image before doing this), if you do this then you 
% can use either metric of "concentration". However spectra often follow power-laws 
% so if you choose this route you should pick 50% or even 90% or 99%. Here is 
% use the first method. I find the smallest number of components which capture 
% X% of the total spectral energy. 
% 
% 
% 
% *frac*=> the fraction of cumulative spectral energy that defines which components 
% to keep
% 
% 
% 
% *imChanCompRatios*  get the compression ratio (minimum number of components 
% containing X% (100*frac%) of the energy divided by the total number of components) 
% for all R G B Gr H S V channels of an RGB image 
% 
% *CompRatByGenYimChan*    apply *imChanCompRatios*to a stack of images 
% 
% *imChanModes*   get the unadulterated spectral energy for all components for 
% all R G B Gr H S V channels of an RGB image  
% 
% *modesByimChanYGen*     apply *imChanModes*to a stack of images
% 
% *sparsifyTransIm*     get only the top components of the transformation on 
% the single channel image provided LOG TRANSFORM the energy of these components 
% and place in uint8 format for viewing
% 
% *transImSparse*     apply *sparsifyTransIm*on all the R G B Gr H S V channels 
% of an RGB image  
% 
% *imByGenSparse*     apply *transImSparse*to a stack of images
% 
% *transImSparseHistEq*     same as *transImSparse* but apply histogram equalization 
% first (make pixel distribution uniform)
% 
% *imByGenSparseHistEq*     apply *transImSparseHistEq*to a stack of images
% 
% *sparsifyTransImDouble*     same as *sparsifyTransIm* but z-score instead 
% of log transform and return in double format for analysis instead of viewing
% 
% *transImSparseDouble*     apply *transImSparseDouble*on all the R G B Gr H 
% S V channels of an RGB image
% 
% *imByGenSparseDouble*     apply *transImSparseDouble*to a stack of images
% 
% *transImSparseHistEqDouble*     same as *transImSparseDouble* but apply histogram 
% equalization first (make pixel distribution uniform)
% 
% *imByGenSparseHistEqDouble*     apply *transImSparseHistEqDouble*to a stack 
% of images
% 
% 
% 
% 

% The two major transforms
applyFFT=@(x) abs(fftshift(fft2(x-mean(x(:))))).^2; % simple FFT - ALWAYs remove the mean first (as demonstrated here)
applyDCT=@(x) dct2(x-mean(x(:))).^2; % simple DCT - ALWAYs remove the mean first (as demonstrated here)
% Gabor wavelet transform
% spatial orientation tree wavelet
%


%%%%% BEGIN SECTION DEFINING THE TRANSFORMATION FUNCTIONS ON THE FLY

% STEP ONE pick one of these to use
transFunc_helper=@(x) x; % do not transform the image
% transFunc_helper=@(x) uint8ify(imdilate(getEdges(imgaussfilt(x,1.5),'Sobel'),strel('disk',1))); % edge detected
% transFunc_helper=@(x) entFilt(x,9); % entropy filter
% transFunc_helper=@(x) uint8ify(supPixSegCombChan(x,128)); %segmented (no texture no edges)
% transFunc_helper=@(x) uint8ify(cast2NGrayLevels(supPixSegCombChan(x,128),3)); %segmented (no texture no edges)

% STEP TWO pick of these
% transFunc=@(x) applyFFT(transFunc_helper(x));
transFunc=@(x) applyDCT(transFunc_helper(x));

% STEP THREE change some hard-coded parameters (explained in next code block)
% % IF USING DCT USE THESE VALUES
rayLength=255;
centerXY=[0,0];
maxRot=90;
% % IF USING FFT USE THESE VALUES


% STEP FOUR run the command eval(recreateFunctionsString) or run the rest
% of the lines of this code block and the next code block

%%%%% END SECTION DEFINING THE TRANSFORMATION FUNCTIONS ON THE FLY


% we need to adjust some parameters automatically depending on the chosen
% function
paramOpts={'rayLength=255; centerXY=[0,0]; maxRot=90;','rayLength=120; centerXY=[128,128]; maxRot=180; '};
eval(paramOpts{1+contains(evalc('transFunc'),'applyFFT')}); % this line sets the parameters in the workspace after detecting the user's choice




% get a record of the functions defined before this point
notTransFuncDep=whos;
notTransFuncDep={notTransFuncDep([strcmp('function_handle',{notTransFuncDep.class})]).name};


% these function depend on transFunc and prepImag

% get compression ratios
imChanCompRatios=@(x,frac) squeeze(cellfun(@(x) compRatio(transFunc(x),frac),prepImag(x)))';
CompRatByGenYimChan=@(imStack,frac) cell2mat(cellfun(@(x) imChanCompRatios(x,frac),cellifyImStack(imStack),'UniformOutput',false));


% get the raw modes
imChanModes=@(x) cell2mat(squeeze(cellfun(@(x) unwrap(transFunc(x))',prepImag(x),'UniformOutput',false)))';
modesByimChanYGen=@(imStack) cell2mat(permute(cellfun(@(x) imChanModes(x),cellifyImStack(imStack),'UniformOutput',false),[2,3,1]));

% get a uint8 of log transformed and rescaled transformation (log rescaling
% not always required)
sparsifyTransIm=@(x,frac) uint8ify(log(x)).*uint8(x>=compThresh(sort(unwrap(x),'descend'),frac));
transImSparse=@(x,frac) cell2mat(cellfun(@(x) sparsifyTransIm(transFunc(x),frac),prepImag(x),'UniformOutput',false));
imByGenSparse=@(imStack,frac) cell2mat(cellfun(@(x) transImSparse(x,frac),permute(cellifyImStack(imStack),[2,3,4,1]),'UniformOutput',false));

% get histogram equalized uint8 log of transformed images
transImSparseHistEq=@(x,frac) cell2mat(cellfun(@(x) sparsifyTransIm(transFunc(histEq(x)),frac),prepImag(x),'UniformOutput',false));
imByGenSparseHistEq=@(imStack,frac) cell2mat(cellfun(@(x) transImSparseHistEq(x,frac),permute(cellifyImStack(imStack),[2,3,4,1]),'UniformOutput',false));

% get the transformation of images without log rescaling or uint8 transform
% (we still subtract the minimum)
minShifted=@(x) x-min(x(:));
rangeNorm=@(x) std(x(:)).*(minShifted(x))./range(x(:));
STDscaled=@(x) std(x(:)).*(rangeNorm(x));
sparsifyTransImDouble=@(x,frac) minShifted(double(x)).*double(x>=compThresh(sort(unwrap(x),'descend'),frac));
transImSparseDouble=@(x,frac) cell2mat(cellfun(@(x) sparsifyTransImDouble(transFunc(x),frac),prepImag(x),'UniformOutput',false));
imByGenSparseDouble=@(imStack,frac) cell2mat(cellfun(@(x) transImSparseDouble(x,frac),permute(cellifyImStack(imStack),[2,3,4,1]),'UniformOutput',false));

% get the histogram equalized transformation
transImSparseHistEqDouble=@(x,frac) cell2mat(cellfun(@(x) sparsifyTransImDouble(transFunc(histEq(x)),frac),prepImag(x),'UniformOutput',false));
imByGenSparseHistEqDouble=@(imStack,frac) cell2mat(cellfun(@(x) transImSparseHistEqDouble(x,frac),permute(cellifyImStack(imStack),[2,3,4,1]),'UniformOutput',false));

%% 
%% 
%% Shape analysis of FFT and DCT
% 
% 
% When we visualize the FFT and DCT contours it will become clear how image 
% features produce either "star shaped" or "blob shaped" FFT and DCT plots. These 
% can be roughly used to categories images (indoor vs outdoor etc.). To quantify 
% these shapes we need contours. MATLAB does have a built in contour finding process 
% which may produce competitive results. However I felt that a more solid foundation 
% would be had if we could precisely parameterize the shapes in terms of radial 
% and angular coordinates and offer fine tuned control over contours. MATLAB's 
% contours will isolate small "islands", this problem is relatively small if using 
% contours after integrating out from the origin in all directions, but it still 
% exists. These islands can also be removed in post processing so a comparison 
% to MATLAB's built in methods is worthwhile but not implemented here. Instead 
% I precisely define rays extending from the origin and integrate along them to 
% find the fractional energy point (*fep*) or find where the ray crosses a global 
% threshold identified by integrating all the points, not just along that ray. 
% Since each ray has a precisely defined angle I can capture the radius and angle 
% of each contour point. 
% 
% The blobiness or star-shapedness of a contour is a function of how many peaks 
% there are in a plot of radius vs angle, as well as the ratio of contour edge 
% length to contour area and the width of those peaks. The shape of the counter 
% sensitively depends on the fraction of energy you want to keep. Furthermore 
% the contours are highly variable across nearly identical images. As a consequence 
% this metric and those derived from them were not especially useful and discriminating. 
% With more stable contour parameterization methods they may nonetheless provide 
% a solid basis for quantifying explorations of fc6 space.
% 
% 
% 
% Related metrics defined in this section include spectral entropy, the compression 
% ratio of the _contour itself_ and the mirror symmetry of the contour. Spectral 
% entropy is a lot like normal entropy only instead of using an actual pdf (probability 
% distribution function) you apply pdf normalization to the energy spectrum of 
% the transformation and "pretend" it's a pdf. Thus spectral entropy is -sum( 
% normSpectra * log(normSpectra)) where "normSpecta" is the vector of pdf normalized 
% spectral energy components. The compression ratio of the contour itself measures 
% how peaky it is and how regular the peaks are. The mirror symmetry is also a 
% measure of how regular the peaks are. However for FFT it also lets us distinguish 
% image features that are rotated by 90 degrees. 
% 
% 
% 
% A very important feature about this code to recognize is that if you change 
% between FFT and DCT you need to change some parameters involved with contour 
% estimation. FFT puts the origin in the center of the plot, therefore the contour 
% extends from 0 to 360 degrees and the ray extends from 0 to 127 pixels in length 
% (I set to 120 to avoid some image resizing problems). DCT puts the origin in 
% one corner, thus the rays are longer (able to get the full 256) but the contour 
% extends only from 0 to 90 degrees (-90 visually, +90 in row-column coordinates). 
% So you need to update the corresponding values *rayLength*, *maxRot* and *centerXY* 
% (*maxRot* is 180 for FFT instead of 360 because the FFT is always mirror symmetric 
% so it's redundant). Examples are provided in the code block below and in the 
% one above (where changing the transform is defined).
% 
% 
% 
% 
% 
% *getRayVals*  get the transform spectrum values of a ray with the specified 
% width and angle and length for the raw spectrum provided (cannot sparsify in 
% advance)
% 
% *getRay*  get a smoothed version of *getRayVals*
% 
% *getRayNdcs*  get the linear indices of the ray found in *getRayVals* (useful 
% for plotting the single ray)
% 
% *getRotRays*  get the *fep*(fractional energy point, AKA contour radius) for 
% all the angles specified in *rotations*
% 
% *getRayContours*  get all the indices of the region interior to the contour 
% (used to plot the entire contour)
% 
% *getRaySmooth*  Use the rays to smooth a full spectrum prior to trying to 
% fit a surface to it (probably doesn't work, I abandoned it but smoothing in 
% polar versus Euclidean coordinates is useful)
% 
% *getSpectralEntropy*  get the spectral entropy of the entire spectrum you 
% provide (useful to limit the spectrum)
% 
% *getPeakStats*  return a large set of stats, including those about the contour 
% and others for the spectrum of a single image channel (you provide the spectrum) 
% and the results of *getRotRays*(which is the variable "x")
% 
% *getRotCorr1*  get the set of stats provided by *getPeakStats* but without 
% you having to provide *getRotRays*
% 
% *rotSymStats*  get the set of stats provided by *getPeakStats* but without 
% you having to provide a spectrum and doing it for all 7 R G B Gr H S V channels 
% of an RGB image you provide
% 
% *rotSymStatsByGenYimChan*  perform *rotSymStats*on a 4D image stack
% 
% 
% 
% 

% Now we define functions associated with capturing countours in the
% transformed space and analyzing their shapes
% Examine angularly distributed rays of the transform

% % IF USING DCT USE THESE VALUES
% rayLength=255;
% centerXY=[0,0];
% maxRot=90;
% % IF USING FFT USE THESE VALUES
% rayLength=120;
% centerXY=[128,128];
% maxRot=180;


rayWidth=8;
raySmoothWindow=8; % smoothing window along the ray
rayFrac=0.97;
NtexturePartitions=128;

compFrac=0.97;
contourPeakPtile=66;

% define some derived parameters
rotations=num2cell([0:0.5:maxRot]);
[Xtrash,Ytrash]=meshgrid(max(1,centerXY(1)-floor(rayLength/2)):1:(centerXY(1)+floor(rayLength/2)),max(1,centerXY(2)-floor(rayLength/2)):1:(centerXY(2)+floor(rayLength/2)));
specEntBand=sub2ind([256,256],Ytrash(:),Xtrash(:)); clearvars('Xtrash','Ytrash');

% utility functions
cropBox=@(x) [ceil((size(x,[1,2])-2*rayLength-1)/2),2*rayLength,2*rayLength];
cropRotIm=@(x)  imcrop(x,cropBox(x));

% get the Frac power point along a ray and distance to that point
rayWidthSpace=@(rayWidth) (-floor(rayWidth/2):floor(rayWidth/2))+1-(rayWidth>=2);
rayPoints=@(len,yNdx) [(1:len)',yNdx*ones(len,1)];
rotMat=@(theta) [[cos(theta),-sin(theta)];[sin(theta),cos(theta)]];
getRaySubs_helper=@(raySubs) [nan(max(sum(raySubs<1,1)),2);raySubs(all(raySubs>=1,2),:)];
getRaySubs=@(len,yNdx,theta,centerXY) getRaySubs_helper(round(rayPoints(len,yNdx)*rotMat(theta)')+centerXY);
getRayNdcs_helper=@(raySubs,imSize) sub2ind(imSize,raySubs(:,2),raySubs(:,1));
getRayNdcs=@(im,rayWidth,rayLen,theta,centerXY) cell2mat(cellfun(@(x) getRayNdcs_helper(getRaySubs(rayLen,x,theta,centerXY),size(im)) ,num2cell(rayWidthSpace(rayWidth)),'UniformOutput',false));
getRayVals_helper=@(im,ray) [nan(sum(isnan(ray)),1);im(ray(~isnan(ray)))];
getRayVals=@(im,rayWidth,rayLen,theta,centerXY)   cell2mat(cellfun(@(x)  getRayVals_helper(im,getRayNdcs_helper(getRaySubs(rayLen,x,theta,centerXY),size(im)))  ,num2cell(rayWidthSpace(rayWidth)),'UniformOutput',false));


getRay=@(imFull,maxLength,x,centerXY,raySmoothWindow) smoothdata(nanmean(getRayVals(imFull,rayWidth,maxLength,x,centerXY),2),'movmean',raySmoothWindow)';

% get the point where the ray crosses a THRESHOLD SPECIFIC TO THAT RAY
% getFracPowerValue_helper=@(ray,fep) fep; %[fep,ray(fep),mean(ray(1:fep))];
% getFracPowerPoint_helper1=@(ray,frac) find((cumsum(ray)./sum(ray))>=frac,1,'first');
% getFracPowerPoint_helper=@(ray,frac) getFracPowerValue_helper(ray,getFracPowerPoint_helper1(ray,frac));
% % getFracPowerRay=@(x,maxLength,frac) getFracPowerPoint_helper(getRay(x,maxLength),frac);
% % getRotRays=@(imFull) cell2mat(cellfun(@(x) getFracPowerRay(cropRotIm(imrotate(imFull,x)),rayLength,0.95),rotations,'UniformOutput',false));
% getRotRays=@(imFull,rayFrac) cell2mat(cellfun(@(x) getFracPowerPoint_helper(getRay(imFull,rayLength,deg2rad(x),centerXY,raySmoothWindow),rayFrac),rotations,'UniformOutput',false));

% get the point where the ray crosses a GLOBAL THRESHOLD
getFracPowerValue_helper=@(ray,fep) fep; %[fep,ray(fep),mean(ray(1:fep))];
getFracPowerPoint_helper1=@(ray,thresh) find(smooth(ray,ceil(numel(ray)/5))<thresh,1,'first');
getFracPowerPoint_helper=@(ray,thresh) getFracPowerValue_helper(ray,getFracPowerPoint_helper1(ray,thresh));
getRotRays=@(imFull,rayFrac) cell2mat(cellfun(@(x) getFracPowerPoint_helper(getRay(imFull,rayLength,deg2rad(x),centerXY,raySmoothWindow),compThresh(imFull,rayFrac)),rotations,'UniformOutput',false));

% get all the points leading up to that threshold crossing
getRayContours_helper2=@(rayNdcs,cutNdx) rayNdcs(1:cutNdx,:);
getRayContours_helper1=@(rayNdcs) rayNdcs(~isnan(rayNdcs(:)));
getRayContours_helper0=@(im,rayWidth,rayLen,theta,centerXY,frac) getRayContours_helper1(getRayContours_helper2(getRayNdcs(im,1,rayLen,theta,centerXY),min(rayLen,1+getFracPowerPoint_helper1(getRay(im,rayLen,theta,centerXY,raySmoothWindow),frac))));
% % RAY SPECIFIC THRESHOLD
% getRayContours=@(imFull,rayFrac) cell2mat(cellfun(@(x) getRayContours_helper0(imFull,rayWidth,rayLength,deg2rad(x),centerXY,rayFrac),rotations','UniformOutput',false));
% % GLOBAL THRESHOLD
getRayContours=@(imFull,rayFrac) cell2mat(cellfun(@(x) getRayContours_helper0(imFull,rayWidth,rayLength,deg2rad(x),centerXY,compThresh(imFull,rayFrac)),rotations','UniformOutput',false));

% apply a radial smoothing to a transform
smoothRots=num2cell(atan([(0:256)./256,256./fliplr(0:256)]));
removeNans=@(x) x(~isnan(x));
getRaySmooth=@(imFull) cell2mat(cellfun(@(x) [removeNans(getRayNdcs(imFull,1,rayLength,x,centerXY)),smooth(removeNans(getRayVals(imFull,1,rayLength,x,centerXY)),raySmoothWindow)],permute(smoothRots,[2,1]),'UniformOutput',false));




% capture a measure of the rotational symmetry of the FFT by actually
% rotating an FFT image
getRotCorr_helper=@(imCropped,imRot) corr(unwrap(imCropped),unwrap(imRot),'type','Spearman');
getRotCorr=@(imCropped,imFull) cellfun(@(x) getRotCorr_helper(imCropped,cropRotIm(imrotate(imFull,x))),rotations);

% get the spectral entropy
spectrumRegularize=@(imSpectrum) min(abs(imSpectrum(imSpectrum(:)~=0))).*(imSpectrum(:)==0); % just lets us avoid putting zeros in a log function
getSpectralEntropy=@(imSpectrum) -sum(imSpectrum(:).*log(imSpectrum(:)+spectrumRegularize(imSpectrum)));

% get a set of stats about the image channels
getPeakStats_helper2 =@(x,biniarized,nRotPeaks) [max([-inf,nRotPeaks]),max([-inf,sum((x-prctile(x,2)).*biniarized)/(sum(x)*nRotPeaks)])];
getPeakStats_helper1=@(x,biniarized) getPeakStats_helper2(x,biniarized,(sum(abs(diff(biniarized)))+sum(biniarized([1,end])))/2);
getPeakStats=@(x,imSpectrum,frac,im)  [getPeakStats_helper1(x,smooth(x,3)'>=prctile(smooth(x,3),contourPeakPtile)),...
    max([-inf,corr(x',fliplr(x)')]),...
    max([-inf,getSpectralEntropy(imSpectrum(specEntBand)./sum(imSpectrum(specEntBand),'all'))]),...
    max([-inf,sum(x)]),...
    max([-inf,sum(abs(diff(x)))]),...
    max([-inf,compRatio(abs(fftshift(fft(zscore(x)))).^2,compFrac)]),...
    max([-inf,compRatio(imSpectrum(:),compFrac)]),...
    max([-inf,getImEntropy(im)]),...
    max([-inf,meanHOGSel(im,cellSize,HOGptile)]),...
    max([-inf,texIndexSepChan(im,NtexturePartitions)]),...
    max([-inf,piqe(im)])];

getRotCorr1=@(imSpectrum,frac,im) getPeakStats(getRotRays(imSpectrum,frac),imSpectrum,frac,im);
rotSymStats=@(x,frac) permute(cell2mat(cellfun(@(x) getRotCorr1(sparsifyTransImDouble(transFunc((x)),1),frac,x),prepImag(x),'UniformOutput',false)),[2,1,3]);
rotSymStatsByGenYimChan=@(imStack,frac) cell2mat(permute(cellfun(@(x) rotSymStats(x,frac),cellifyImStack(imStack),'UniformOutput',false),[2,1,4,3]));


% capture a list of all functions that depend on the transFunc and a string
% we can use to recreate these functions any time we change the transFunc
transFuncDep={'imChanCompRatios','CompRatByGenYimChan','imChanModes','modesByimChanYGen','sparsifyTransIm','transImSparse','imByGenSparse','transImSparseHistEq','imByGenSparseHistEq',...
    'minShifted','rangeNorm','STDscaled','sparsifyTransImDouble','transImSparseDouble','imByGenSparseDouble','transImSparseHistEqDouble','imByGenSparseHistEqDouble','cropBox',...
    'cropRotIm','rayWidthSpace','rayPoints','rotMat','getRaySubs_helper','getRaySubs','getRayNdcs_helper','getRayNdcs','getRayVals_helper','getRayVals','getRay','getFracPowerValue_helper',...
    'getFracPowerPoint_helper','getRotRays','getRayContours_helper2','getRayContours_helper1','getRayContours_helper0','getRayContours','removeNans','getRaySmooth','getRotCorr_helper',...
    'getRotCorr','spectrumRegularize','getPeakStats_helper2','getPeakStats_helper1','getPeakStats','getRotCorr1','rotSymStats','rotSymStatsByGenYimChan'}; % the list MUST be in the order the functions are defined

functionsStringPrefix='rotations=num2cell([0:0.5:maxRot]); [Xtrash,Ytrash]=meshgrid(max(1,centerXY(1)-floor(rayLength/2)):1:(centerXY(1)+floor(rayLength/2)),max(1,centerXY(2)-floor(rayLength/2)):1:(centerXY(2)+floor(rayLength/2))); specEntBand=sub2ind([256,256],Ytrash(:),Xtrash(:)); clearvars(''Xtrash'',''Ytrash'');';
recreateFunctionsString={};
for ndx=1:length(transFuncDep)
    if ~isempty(whos(transFuncDep{ndx}))
        temp=strsplit(evalc(transFuncDep{ndx}),newline);
        recreateFunctionsString{end+1}=[temp{2},temp{4}];
    end
end
recreateFunctionsString=[strjoin(recreateFunctionsString,'; '),';'];
recreateFunctionsString=[functionsStringPrefix,recreateFunctionsString];
evalCount=0;
recreateFunctionsString=['evalCount=evalCount+1; eval(paramOpts{1+contains(evalc(''transFunc''),''applyFFT'')}); clearvars(transFuncDep{:});',recreateFunctionsString];

% at any point now you can redefine transFunc and use the command
% "evalc(recreateFunctionsString)" to propagate your changes through to all
% the other functions