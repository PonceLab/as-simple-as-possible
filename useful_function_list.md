All statistical tests of significance functions were native to Matlab and included signrank.m, ranksum.m, anova1.m, kruskalwallis.m, randperm.m, randsample.m, datasample.m and bootstrp.m. Normalization was performed via zscore.m. In order to compute the Wilcoxon rank sum effect size as a simple difference (Kerby, 2014, The Simple Difference Formula), we used the customized function ranksum2.m.

Results section “Neurons responded maximally to natural images”

- To fit the receptive field data, we used fmgaussfit.m (Nathan Orloff, Mathworks Central).

- To label the natural images, we adapted this tutorial at the googleapis/python-vision repo:
https://github.com/googleapis/python-vision/blob/8374f79ca1d7048cbb1638d759557ec9a7f92c6a/samples/snippets/quickstart/quickstart.py. Labels were visualized via the native Matlab function wordcloud.m.

To compute the orientation dominance index, we used the custom function totalHOGSelStack.m

Section 2. “Neurons across the ventral stream led to prototypes”

- The image generator has been published at https://github.com/willwx/XDream and the repository includes additional instructions for running the synthesis experiments.

- As a search algorithm, we used CMA-ES, implemented by Binxu Wang in a custom Matlab function cmaes_simple.m. 

Section 3. “Prototypes in anterior cortex resemble specific object parts. ”

- To compute the time to convergence (“V1/V2 sites required fewer cycles to create their prototypes compared to IT sites…”), we used movmean.m to smooth each site’s mean response per generation curve. 

- To compute image specificity and thus corroborate our interpretation of the convergence rates (“We interpreted these differences in convergence rates as the result of an increase in feature specificity...”), we used the function (@jojker).

- The semantic-ensemble analyses all relied on Matlab’s alexnet.m function.

Section 4. “Prototype complexity comparable to segmented object parts”

- To measure the complexity ratio, we used the custom function compRatio.m, which relies on the native Matlab function dct2.m.

- To measure the number of parts per image (segmentations), we used the function Ms2.m from the package “k-means, mean-shift and normalized-cut segmentation” by Alireza A, from the Mathworks File Exchange (https://www.mathworks.com/matlabcentral/fileexchange/52698-k-means-mean-shift-and-normalized-cut-segmentation).

Section 5. “Prototypes are associated with salient features in natural scenes”

- Image cropping was done using the native Matlab function imcrop.m. Patches were propagated into the native AlexNet, and correlational/cosine distances measured using pdist2.m (also Matlab-native).
