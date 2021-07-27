# As simple as possible
This is a repo expositing methods published in: _As simple as possible, but not simpler: features of the neural code for object recognition_ authored by  Rose O., Johnson J.K., Wang B. and Ponce C.R.* and published in _To Be Determined_  Month 2021

We demonstrate the methods here in a manner that researchers familiar with MATLAB and Python can easily read, understand, and adapt to their purposes. Please cite as:
xxx

## Table of contents
This repo is organized into four folders corresponding the class of analysis plus two folders for data and utility functions needed for all folders. In each folder is one or more livescripts which take readers through toy examples and show them how to adapt them to their own needs. Each live script is also converted to markdown for display as the readme for that folder. 

- [Image Statistic Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Statistic_Analysis/README.md): Orientation dominance index, Entropy, Energy, Stationarity, Luminance, Tamura's textures
- [Semantic Ensemble Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Semantic_Ensemble_Analysis/README.md): How to create a curvature ensemble from a toy data set and apply it to several images.
- [Complexity Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Complexity_Analysis/README.md): How to analyze the number of image parts after pixel-space clustering and get the complexity ratio
- [Image Content Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis/README.md): How to get fuzzy-labels for images with a resnet trained for the COCO-STUFF data set and then gathering statistics on word frequency in labels, including Google Cloud Vision labels
	- [Cloud Vision Labeling](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis/labeling_with_Google_Cloud_Vision.md): How to get labels from the Google Cloud Vision API. 

## How to use this repo

The links in the table of contents take you to markdown files for display on github. We also provide actual MATLAB livescripts in the same folders. Simply clone this repository to your computer and browse the directories. Whenever possible we include dependencies here in this repository. When we do not we offer instructions on installing them yourself in the livescripts where they are needed. If a function is not recognized then it is probably in a [standard MATLAB toolbox](https://www.mathworks.com/help/matlab/index.html) (e.g. the [computer vision toolbox](https://www.mathworks.com/products/computer-vision.html)). We may not have noticed since we install all toolboxes by default on our laboratory machines. 

Clone the repo, get the livescripts running, and study the examples. Then adapt for your own purposes. Just don't forget to cite us please!

## Related repositories

A neuron-guided image synthesis algorithm called [Xdream](https://github.com/willwx/XDream) was developed in collaboration with Carlos Ponce. [Binxu Wang](https://github.com/Animadversio) replaced the optimizer with [CMAES](https://github.com/Animadversio/CMAES_optimizer_matlab) and adapted the [image generating up-convolution CNN](https://github.com/Evolving-AI-Lab/synthesizing) for MATLAB (to be released in an upcoming publication). 

Should you like to share your MATLAB livescripts on Github you may be interested in this function which automatically converts livescripts into markdown so they can be displayed like jupyter notebooks: [latex2markdown](https://blogs.mathworks.com/pick/2021/01/08/convert-your-live-scripts-to-markdown-file/)


### Update 07/16/2021
Work was completed on 07/06/2021 the remaining work is to:
- [ ] Get scrutiny from the other authors
- [x] Link to stuff that carlos recommended
- [x] Show how to install the alexnet for matlab stuff (reference things I shared with victoria and katie) 
- [x] check for toolbox dependencies
- [x] Update this main readme to provide guidance for downloading and running


### ~~Assignments for tasks within the PonceLab group~~
~~Work date: 07/06/2021~~

~~Guidelines: Everyone make a fork or just download the structure work on your computer then James will do the merge. Be mindful of what should probably go in the "utils" and "data" folder and do that. Keep the analysis folders limited to the livescript + any scripts or functions that absolutely could not fit in the livescript, and that no one else is likely to use in their folders. Thus, ranksum2 would go in utils, not in a livescript. All dependencies should be in either the analysis folder or the utils function unless they are owned by someone else (e.g. a FEX file, or the resnet repo etc.). For dependencies that are not able to be included, please directly list them at the top of the livescript.~~

~~CRP: 
Fill out the complexity code folder, and the ensembles folder~~

~~JKJ:
Fill out the Image Statistic Analysis folder and the COCO-stuff part of the image content analysis folder~~

~~WB:
Fill out the Google cloud vision part of the image content analysis folder (+ something about troubleshooting Anaconda and Matlab?)~~

~~After the work date:
James will do the merges, any cleaning that is needed and make sure everything imports like it should. He will put the dependents in a requirements.txt file and/or a script that can be run, or quality advice given to setting up MATLAB to work with python.~~

~~useful links~~

~~https://blogs.mathworks.com/pick/2021/01/08/convert-your-live-scripts-to-markdown-file/~~

~~https://www.markdownguide.org/cheat-sheet/~~
