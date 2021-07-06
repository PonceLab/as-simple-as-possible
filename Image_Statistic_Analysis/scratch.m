function Project_Selectivity_reportOrientationDominance(Monkeys)
% Purpose: This script examines whether there are dominant edges in some
% images vs. others. Specifically testing if reference images (natural and
% artificial) preferred by V1/V2 neurons have more dominant orientations
% compared to IT neurons.

%%





getHOGStats_helper=@(im,cellSize) reshape(extractHOGFeatures(im,'CellSize',[cellSize,cellSize],'BlockSize',[1,1]),9,prod(floor(size(im,[1,2])/cellSize)));
getHOGStats_helperNoZeroMin=@(HOG) HOG+100*max(HOG(:)).*(HOG==0);
getHOGStats_helperSel=@(HOG) (max(HOG,[],1)-min(getHOGStats_helperNoZeroMin(HOG),[],1))./(max(HOG,[],1)+min(getHOGStats_helperNoZeroMin(HOG),[],1));
totalHOGSel_helper=@(HOG,selVec,ptile) sum(HOG(:,selVec>=prctile(selVec,ptile)),2)./sum(HOG(:,selVec>=prctile(selVec,ptile)),'all');
totalHOGSel_helper1=@(HOG,ptile) totalHOGSel_helper(HOG,getHOGStats_helperSel(HOG),ptile);
totalHOGSel=@(im,cellSize,ptile)  getHOGStats_helperSel(totalHOGSel_helper1(getHOGStats_helper(im,cellSize),ptile));
cellifyImStack=@(imStack) squeeze(mat2cell(imStack,size(imStack,1),size(imStack,2),size(imStack,3),ones(1,size(imStack,4))));
applyFunc2Stack2=@(imStack,input2,func4stack) cell2mat(permute(cellfun(@(x) func4stack(x,input2),cellifyImStack(imStack),'UniformOutput',false),[2,3,4,1]));
totalHOGSelStack=@(imStack,cellSize,ptile) applyFunc2Stack2(imStack,{cellSize,ptile},@(im,x) totalHOGSel(im,x{1},x{2}));

myPaths = Project_Selectivity_myPaths ;

array_sets = {33:48 49:64 1:32};
array_loc = {'V1-V2','V4','PIT'} ;
array_colors = ['r','g','b'];

domStat_acrossExp.v1 = [] ;
domStat_acrossExp.v4 = [] ;
domStat_acrossExp.it = [] ;
domStat_acrossExp.dist2rf = [] ; 

LoadStatsFuncs % James' stats

clc

% show examples
imgs_ref = Project_General_returnTestImages(0); % copyright-free images
oriDomStat = squeeze( totalHOGSelStack(imgs_ref,cellSize,90) );

figDom = figure ; 

figure; tiledlayout('flow','TileSpacing','None','Padding','None');
figPos([1000 800 1250 132]);
[~,iSort] = sort(oriDomStat,'descend');
for iPic = 1:length(iSort)
    nexttile
    imshow(imgs_ref(:,:,:,iSort(iPic)))
    title(sprintf('%1.2f',oriDomStat(iSort(iPic))))
    drawnow
end
figPaper([18 2], myPaths.selectivity_general, 'dominant orientation HOG');

mean_dom_across_monks = [] ;
for iM = 1:length(Monkeys)
    M = Monkeys{iM};
    
    for iExp = 1:length(M.Sel)
        fprintf('%d ',iExp); if ~mod(iExp,10); fprintf('\n'); end
        tSel = M.Sel{iExp} ;
        
        %                 CHANNEL RESPONSES
        %           ONLY FOCUSING ON LIVE CHANS PER RF EXPERIMENTS
        iLive =  ismember( tSel.spikeID , M.live_chans ) ;
%         fprintf('\t\t there were %d live sites in the experiment\n ',size( tSel.TunCurve_chans(iLive) ,1))

        %           COLLECT DESCRIPTIONS OF PREFERRED IMAGES
        myDS = imageDatastore( tSel.picLoc ) ;
        [~,picNamesInDS,ext] = cellfun(@fileparts,myDS.Files,'Uni',false) ;
        
        %           OF COURSE, ONLY IF ITS RF WAS CLOSE TO THE LOCATION OF
        %           STIMULUS PRESENTATION
        siteCount = 1 ;
        tSel.prefImages = [] ;
        tSel.prefImages_channelID = [] ;
        tSel.prefImages_rfStim_dist=[] ;
        for iSite = 1:length(tSel.TunCurve_chans)
            
            tSite = tSel.spikeID(iSite) ;
            
            % live channel or not? V1 or not?
            if ~ismember( tSite, M.live_chans )
                continue
            end
            
            % how far was stimulus from RF?
            iSite_inRF = M.RF.chans_unique == tSite ;
            rf_xy = M.RF.chans_center( iSite_inRF , : , 1)  ;
            
            % for each position tested
%             if size( tSel.TunCurve_xy ,1) > 1, keyboard; end
            
            for jxy = 1:size( tSel.TunCurve_xy ,1)
                stim_xy = tSel.TunCurve_xy( 1 , : ) ;
                d = sqrt(sum((rf_xy - stim_xy).^2)) ;

                % identify the images it liked the most
                [~,iSort] = sort( M.Sel{iExp}.TunCurve(iSite,:,1,jxy) ,'descend' ) ;
                top_pic_name_thisChan = M.Sel{iExp}.TunCurve_pics(iSort(1:4)) ;
                
                S_in_DS = find( ismember(picNamesInDS , top_pic_name_thisChan ) ) ;
                imgs = [];
                for ipic = 1:length(S_in_DS)
                    img = readimage(myDS,S_in_DS(ipic));
                    img = imresize(img,[200 200]);
                    if size(img,3) < 3
                        img = repmat(img,[1 1 3]);
                    end
                    
                    imgs = cat(4,imgs,img);
                end
                tSel.prefImages{siteCount} = imgs;
                
                tSel.prefImages_channelID = cat(2,tSel.prefImages_channelID, ...
                repmat(tSite,1,size(imgs,4) ) ) ;
            
                tSel.prefImages_rfStim_dist = cat(2,tSel.prefImages_rfStim_dist,...
                repmat(   d,1, size(imgs,4) ) ) ; 
            
                siteCount = siteCount + 1 ;
                
            end % of jxy
            
        end % of iSite

        topPics = cell2mat( reshape( tSel.prefImages, [1 1 1 length(tSel.prefImages)] ) ) ;
        oriDomStat = squeeze( totalHOGSelStack(topPics,cellSize,90) );
        
        isV1 = ismember( tSel.prefImages_channelID , 33:48 ) & tSel.prefImages_rfStim_dist < 1 ;
        isV4 = ismember( tSel.prefImages_channelID , 49:64 ) & tSel.prefImages_rfStim_dist < 1 ;
        isIT = ismember( tSel.prefImages_channelID , 1:32 )  & tSel.prefImages_rfStim_dist < 1 ;
        
        domStat_acrossExp.v1 = cat(1,domStat_acrossExp.v1,oriDomStat(isV1)) ;
        domStat_acrossExp.v4 = cat(1,domStat_acrossExp.v4,oriDomStat(isV4)) ;
        domStat_acrossExp.it = cat(1,domStat_acrossExp.it,oriDomStat(isIT)) ;
        
    end % of iExp
    fprintf('\n');
    
    
    bootsV1 = bootstrp(500,@median, domStat_acrossExp.v1);
    bootsV4 = bootstrp(500,@median, domStat_acrossExp.v4);
    bootsIT = bootstrp(500,@median, domStat_acrossExp.it);
    [p,~,ss] = ranksum2(domStat_acrossExp.v1 , domStat_acrossExp.it , 'tail','right')  ; 
    
    fprintf('Monkey %s, V1/V2 %1.2f%s%1.2f, V4 %1.2f%s%1.2f and IT %1.2f%s%1.2f (P = %1.1e, r_sdf %1.2f, N_V1 = %d, N_V4 = %d, N_IT = %d)\n',...
        M.monkey,...
        mean(bootsV1), char(177),std(bootsV1),...
        mean(bootsV4), char(177),std(bootsV4),...
        mean(bootsIT), char(177),std(bootsIT), p, ss.simple_difference,...
        numel(domStat_acrossExp.v1),numel(domStat_acrossExp.v4),numel(domStat_acrossExp.it))
    
    mean_dom_across_monks = cat(1,mean_dom_across_monks,... 
                             cat(3,[mean(bootsV1) mean(bootsV4) mean(bootsIT)],...
                                   [std(bootsV1) std(bootsV4) std(bootsIT)] ) ) ; 
    
end % of iM

    figure(figDom)
    ploterr(1:3,mean_dom_across_monks(1,:,1),[],...
                mean_dom_across_monks(1,:,2),'r')
    hold on
    plot(1:3,mean_dom_across_monks(1,:,1),'ro')
    ploterr(1:3,mean_dom_across_monks(2,:,1),[],...
                mean_dom_across_monks(2,:,2),'k')
    plot(1:3,mean_dom_across_monks(2,:,1),'ko')
    
end

