if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

configPath=".\configs\cocostuff164k.yaml";
modelPath="C:\Users\ponce\Desktop\CarlosSetUpFilesHere\CompressionPaperReviewResponse\resources\deeplab-pytorch-master\data\models\coco\deeplabv1_resnet101\caffemodel\deeplabv2_resnet101_msc-cocostuff164k-100000.pth";
imagePath="image.jpg";
cuda=false;
crf=true;
sizeThresh=1/9;
maxIteration=10;
doPlot=false;
[a]=py.singleHierarchy.singleHierarchy(configPath,modelPath,imagePath,cuda,crf,sizeThresh,maxIteration,doPlot);


a=cell(a);
labelMapList=cellfun(@(x) double(x), cell(a{1}),'Uni',false); labelMapList=cat(3,labelMapList{:});
probMapList=cellfun(@(x) double(x), cell(a{2}),'Uni',false);  probMapList=permute(cat(4,probMapList{:}),[2,3,4,1]); 
classList=a{3};
a=[];



% get a list of all the labels represeneted
unqLabels=unique(labelMapList(:)); 
unqClassNames=cellfun( @(x) char(classList{x}),num2cell(unqLabels),'Uni',false);
classList=[];
labelEdges=(unqLabels(1:end-1)+unqLabels(2:end))/2; labelEdges=[unqLabels(1);labelEdges;unqLabels(end)];
% step through the label maps
nLayers=size(labelMapList,3);
segImSize=size(labelMapList,[1,2]);
sortedLabels=[];
sortedClassNames={};
labelLayer=[];
unWrap=@(x) x(:);
for layer=1:nLayers
% keep the unique values which cover more than 1/9th of the image
N=histcounts(unWrap(labelMapList(:,:,layer)),labelEdges,'Normalization','probability');
bigLabels=unqLabels(N>=sizeThresh);
bigClassNames=unqClassNames(N>=sizeThresh);
% sort them by size and keep track of what level they are in
% append them to the previous level
[~,sortNdcs]=sort(N(N>=sizeThresh),'descend');
labelLayer=[labelLayer;layer*ones(size(bigLabels))];
sortedLabels=[sortedLabels;bigLabels(sortNdcs)];
sortedClassNames=[sortedClassNames;bigClassNames(sortNdcs)];
end

nLabels=numel(sortedLabels);
% make a blank adjacency matrix
labelNetworkOverlap=zeros(nLabels);
labelNetworkOverlapDir=zeros(nLabels);
labelNetworkJointProb=zeros(nLabels);
labelNetworkRelEnt=zeros(nLabels);
% make a deep boolean matrix representing the locations of all the labels represented (in order of the list)
labelLocs=false([size(labelMapList,[1,2]),numel(sortedLabels)]);
probMap=squeeze(probMapList(:,:,1,:)); probMap=probMap./sum(probMap,[1,2]);
labelLocMap=zeros(size(labelMapList,[1,2])); % the sums in this map are uniquely defined by a combination of other layers
sumLabelCodes=ones(1,nLabels); 
for labelNdx=1:nLabels
    label=sortedLabels(labelNdx);
    labelLocs(:,:,labelNdx)=any(labelMapList==label,3);
    labelLocMap(labelLocs(:,:,labelNdx))=labelLocMap(labelLocs(:,:,labelNdx))+sumLabelCodes(labelNdx);
    if labelNdx<nLabels
    sumLabelCodes(labelNdx+1)=sum(sumLabelCodes(1:labelNdx))+1;
    end
end
% step through the labels and make the strength of the connection equal to the overlap of sections
% we could also use joint distribution metrics (KL-divergence, mutual information, etc)
for rowNdx=1:nLabels
    for colNdx=1:nLabels
        if rowNdx==colNdx; continue; end
        if rowNdx>=colNdx
        labelNetworkOverlap(rowNdx,colNdx)=sum(labelLocs(:,:,rowNdx) & labelLocs(:,:,colNdx) ,'all')./sum(labelLocs(:,:,rowNdx) | labelLocs(:,:,colNdx) ,'all');
        labelNetworkOverlap(colNdx,rowNdx)=labelNetworkOverlap(rowNdx,colNdx);
        labelNetworkJointProb(rowNdx,colNdx)=sum(exp(log(probMap(:,:,rowNdx))+log(probMap(:,:,colNdx))),'all'); % prevents some round off errors
        labelNetworkJointProb(colNdx,rowNdx)=labelNetworkJointProb(rowNdx,colNdx);
        end
        labelNetworkOverlapDir(rowNdx,colNdx)=sum(labelLocs(:,:,rowNdx) & labelLocs(:,:,colNdx) ,'all')./sum(labelLocs(:,:,rowNdx),'all');
        labelNetworkRelEnt(rowNdx,colNdx)=sum(probMap(:,:,rowNdx).*(log(probMap(:,:,rowNdx))-log(probMap(:,:,colNdx))),'all');
    end
end
labelNetworkOverlap(logical(eye(size(labelNetworkOverlap))))=1;
% visualize the network
graphObj=graph(labelNetworkOverlap,sortedClassNames,'omitselfloops');
LWidths = 5*graphObj.Edges.Weight/max(graphObj.Edges.Weight);
% visualize by tying each label to it's segment centroid (avoid collisions)
% step through the labels and get their probability weighted locations
[cols,rows]=meshgrid(1:segImSize(2),1:segImSize(1));
labelCentroids=zeros(nLabels,2);
for labelNdx=1:nLabels
    label=sortedLabels(labelNdx);
    layer=labelLayer(labelNdx);
    probMap=probMapList(:,:,1,label+1);
    probMap=probMap(:)./sum(probMap(:));
    labelCentroids(labelNdx,:)=round(sum([rows(:).*probMap,cols(:).*probMap],1));
end
figure
subplot(1,2,1) 
p=plot(graphObj,'LineWidth',LWidths,'Layout','force','WeightEffect','inverse','XStart',labelCentroids(:,2),'YStart',labelCentroids(:,1));
title('image word cloud')

subplot(1,2,2) % plot the main clusters
numClusts=min(2,sum(labelLayer==1));
[splits]=cluster(linkage(squareform(1-labelNetworkOverlap),'average'),'maxclust',numClusts);
numClusts=numel(unique(splits));

originalImage=imresize(imread(imagePath),size(labelLocMap,[1,2]));
imSize=size(originalImage);
clustMasks=zeros([size(labelLocMap,[1,2]),numClusts]);
clustClouds=cell(1,numClusts);
clustIms=cell(1,numClusts);
for ndx=1:numClusts
    clustMasks(:,:,ndx)=sum(labelLocs(:,:,splits==ndx),3);
    clustClouds{ndx}=sortedClassNames(splits==ndx);
    heatMapOverlay=labeloverlay(originalImage,clustMasks(:,:,ndx),'Colormap','parula');
    FontSize=round(32+32/length(clustClouds{ndx}));
    % make sure the text fits in the image
    fontWidth=FontSize/2;
    imWidth=ceil(imSize(2)/fontWidth);
    cloudString=strjoin(clustClouds{ndx},', ');
    if length(cloudString)>=imWidth;
        nParts=ceil(length(cloudString)/imWidth);
    newLineLocs=zeros(1,nParts);
        temp=strfind(cloudString,' ');
        for insertCnt=1:nParts
            if insertCnt==1
            newLineLocs(insertCnt)=temp(find(temp<((insertCnt-1)*(imWidth-1)+imWidth) & temp>((insertCnt-1)*(imWidth-1)+1),1,'last'));
            else
            newLineLocs(insertCnt)=temp(find(temp<(newLineLocs(insertCnt-1)+imWidth) & temp>newLineLocs(insertCnt-1),1,'last'));
            end
        end
        cloudString(newLineLocs)=newline;
    end
    clustIms{ndx}=insertText(heatMapOverlay,[0,0],cloudString,'BoxOpacity',0,'TextColor',[1,0.97,0.95],'FontSize',FontSize-2);
end
montage(clustIms)




nLabels=numel(sortedLabels);
sumLabelCodes=ones(1,nLabels); 
for labelNdx=1:nLabels
    if labelNdx<nLabels
    sumLabelCodes(labelNdx+1)=sum(sumLabelCodes(1:labelNdx))+1;
    end
end
factorList=cell(nLabels,1);
for labNdx=1:nLabels
    factorList{labNdx}=factor(sumLabelCodes(labNdx));
end


isOdd=mod(labelLocMap,2)==1;
labelLocMap2=labelLocMap;
labelLocMap2(isOdd)=max(1,labelLocMap2(isOdd)-1);
labelLocs2=(labelLocMap2==permute(sumLabelCodes,[1,3,2]));
labelLocs2(:,:,1)=labelLocs2(:,:,1)  |  isOdd ;
uniqueSums=unique(labelLocMap2(:));
uniqueSums=setdiff(uniqueSums,sumLabelCodes);
uniqueSums=sort(uniqueSums,'descend');
sumComps=cell(1,nSums);

nSums=numel(uniqueSums);
foundsums=[];
missingSums=uniqueSums;
for sumNdx=1:nSums
    if ~isempty(sumComps{sumNdx}); continue; end
    sumVal=uniqueSums(sumNdx);
    
    % see if the sum has only one component
    knownComponents=sumLabelCodes(sumLabelCodes==sumVal);
    if isempty(knownComponents)
        possibleComponents=sort(sumLabelCodes(sumLabelCodes>1 & sumLabelCodes<sumVal),'descend');
        if sum(possibleComponents)==sumVal % see if the sum is all possible components
            knownComponents=possibleComponents;
        else
            % test all possible combinations
            compRows=possibleComponents'; % the first combination is just the one possible component so the "component rows" is width==1
            sumTest=sum(compRows,2); 
            cnt=0;
            while ~any(sumTest==sumVal)
                cnt=cnt+1;
                % make a matrix of the sum of prior combinations  with another set of possible components
                [rows,cols]=meshgrid(1:size(compRows,1),possibleComponents);
                sumTest=possibleComponents'+sumTest'; % here is the matrix
                indexer=sumTest<=sumVal; % get rid of values larger than the sum
                compRows=sort([compRows(rows(indexer),:),cols(indexer)],2); % list the expanded set of combinations we just tested
                sumTest=sumTest(indexer);
                [compRows,indexer]=unique(compRows,'rows'); % the order of components doesn't matter so keep unique rows
                indexer=indexer(~any(diff(compRows,[],2)==0,2)); % no component can be repeated 2 or more times so get rid of repeats
                compRows=compRows(~any(diff(compRows,[],2)==0,2),:);
                sumTest=sumTest(indexer);
                % we might not have found the combination for the sum we are checking but we might have stumbled across another
                checkSums=sumTest==missingSums';
                if any(checkSums,'all')
                    foundSumNdcs=setdiff(find(any(checkSums,1)),foundsums);
                    foundsums=unique([foundSumNdcs,foundsums]);
                    missingSums(foundsums)=0;
                    disp(['found ',num2str(length(foundSumNdcs)),' other sums'])
                    for foundSumNdx=foundSumNdcs
                            if ~isempty(sumComps{foundSumNdx}); continue; end
                        sumComps{foundSumNdx}=compRows(sum(compRows,2)==uniqueSums(foundSumNdx),:);
                        if size(sumComps{foundSumNdx},1)~=1; asdfadsfsafds; end
                    end
                end
                if cnt>(nLabels/2)
                    [cnt,sumNdx]
                end
            end
            knownComponents=compRows(sum(compRows,2)==sumVal,:);
            if size(knownComponents,1)~=1; asdfadsfsafds; end
        end
    end
            sumComps{sumNdx}=knownComponents;
    sumNdx
end

for sumNdx=1:nSums
    knownComponents=sumComps{sumNdx};
    sumVal=uniqueSums(sumNdx);
    labelLocs2(:,:,ismember(sumLabelCodes,knownComponents))=labelLocs2(:,:,ismember(sumLabelCodes,knownComponents)) | labelLocMap2==sumVal;
end

im=imresize(imread(imagePath),imSize(1:2));

    figure
    for ndx=1:nLabels; imshow(imoverlay(im,~labelLocs2(:,:,ndx))); title([num2str(ndx),' ',sortedClassNames{ndx}]); drawnow; pause(1); end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





