# As simple as possible
This is a repo expositing methods published in: _Visual prototypes in the ventral stream are attuned to complexity and gaze behaviour_ authored by  Rose O., Johnson J.K., Wang B. and Ponce C.R.* and published in _Nature Communications_  (date TBA, end of 2021)

We demonstrate the methods here in a manner that researchers familiar with MATLAB and Python can easily read, understand, and adapt to their purposes. Please cite as:
xxx

In the paper we used a neuron-guided image synthesis approach to generate the images which evoke the maximum amount of action-potentials at electrode sites. This shows us what types of images trigger the neurons at these sites to work the hardest, thus illuminating what these neurons "encode". A key principle is that only neuron-guided image synthesis can minimize the artificial constraints placed on stimuli used to test neural activity. Experimentors who do not minimize constraints can only interrogate neural activity as it relates to the expectations which guide their choice of stimuli. We found new things, namely that our images evoked stronger activity than both natural images and traditional stimulus sets and that these images had an intermediate level of visual complexity while still recapitulating the overall progression of feature-selectivity from early to late visual areas. 

We also demonstrate several important practices unique to the use of neuron-guided image synthesis. We showed how to formulate traditional hypothesis about visual selectivity such that agreement with those hypothesis could be scored with convolutional neural networks operating on the synthesized images. We showed how to quantitavely compare the semantic content of two or more image sets and demonstrate that differences in activity were due to image content and not confounding texture-like (adversarial) factors. Lastly we showed how to get an appraisal of what sorts of things an abstract image looks-like that is unbiased by experimentor expectations. We hope this work can serve as a model for other researchers applying neuron-guided image synthesis and make this powerful new approach widespread. 

The title of this repo is in reference to one of our central findings that highlights the strength of using neuron-guided image synthesis instead of using hand-curated sets of visual stimuli to probe neural responses. We were able to show that these hand-curated sets are too simplistic. The images the neurons synthesized had higher part-complexity and reconstruction-complexity than tradition stimulus sets but less complexity that photographs, and less than our image synthesis algorithm is capable of making. This implies that the features these neurons are processing must be both rich and sufficiently flexible to allow various kinds of patterns to be decoded, but not so rich that it overfits incidental features of the visual world; to paraphrase, the code must be “as simple as possible, but not simpler.” ~Roger sessions

## Table of contents
This repo is organized into four folders corresponding the class of analysis plus two folders for data and utility functions needed for all folders. In each folder is one or more livescripts which take readers through toy examples and show them how to adapt them to their own needs. Each live script is also converted to markdown for display as the readme for that folder. 

- [Image Statistic Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Statistic_Analysis/README.md): Orientation dominance index, Entropy, Energy, Stationarity, Luminance, Tamura's textures
- [Semantic Ensemble Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Semantic_Ensemble_Analysis/README.md): How to create a curvature ensemble from a toy data set and apply it to several images.
- [Complexity Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Complexity_Analysis/README.md): How to analyze the number of image parts after pixel-space clustering and get the complexity ratio
- [Image Content Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis/README.md): How to get fuzzy-labels for images with a resnet trained for the COCO-STUFF data set and then gathering statistics on word frequency in labels, including Google Cloud Vision labels
	- [Cloud Vision Labeling](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis/labeling_with_Google_Cloud_Vision.md): How to get labels from the Google Cloud Vision API. 

The data are shared via public OSF repo. Image files are witheld but will be made available to researchers following the procedures outlined in the journal article. The link to the repository is here: [osf.io/z6gv2/](https://osf.io/z6gv2/)

## How to use this repo

The links in the table of contents take you to markdown files for display on github. We also provide actual MATLAB livescripts in the same folders. Simply clone this repository to your computer and browse the directories. Whenever possible we include dependencies here in this repository. When we do not we offer instructions on installing them yourself in the livescripts where they are needed. If a function is not recognized then it is probably in a [standard MATLAB toolbox](https://www.mathworks.com/help/matlab/index.html) (e.g. the [computer vision toolbox](https://www.mathworks.com/products/computer-vision.html)). We may not have noticed since we install all toolboxes by default on our laboratory machines. 

Clone the repo, get the livescripts running, and study the examples. Then adapt for your own purposes. Just don't forget to cite us please!

## Related repositories

A neuron-guided image synthesis algorithm called [Xdream](https://github.com/willwx/XDream) was developed in collaboration with Carlos Ponce. [Binxu Wang](https://github.com/Animadversio) replaced the optimizer with [CMAES](https://github.com/Animadversio/CMAES_optimizer_matlab) and adapted the [image generating up-convolution CNN](https://github.com/Evolving-AI-Lab/synthesizing) for MATLAB (to be released in an upcoming publication). 

Should you also like to share your MATLAB livescripts on Github you may be interested in this function which automatically converts livescripts into markdown so they can be displayed like jupyter notebooks: [latex2markdown](https://blogs.mathworks.com/pick/2021/01/08/convert-your-live-scripts-to-markdown-file/)
