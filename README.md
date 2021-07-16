# As simple as possible
This is a repo expositing methods used for the publication: _As simple as possible, but not simpler: features of the neural code for object recognition_ authored by  Rose O., Johnson J.K., Wang B. and Ponce C.R.* and published in _To Be Determined_  Month 2021

We demonstrate the methods here in a manner that researchers familiar with MATLAB and Python can easily read, understand, and adapt to their purposes. If these methods are used, in whole, part, or after adaptation please cite:
xxx

XXX License here


## Table of contents
This repo is organized into four folders corresponding the class of analysis plus two folders for data and utility functions needed for all folders. In each folder is one or more livescripts which take readers through toy examples and show them how to adapt them to their own needs. Each live script is also converted to markdown for display as the readme for that folder. 

- [Image Statistic Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Statistic_Analysis): Orientation dominance index, Entropy, Energy, Stationarity, Luminance, Tamura's textures
- [Semantic Ensemble Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Semantic_Ensemble_Analysis): How to create an xxx ensemble from a toy data set (or maybe the actual data set used in the paper?)
- [Complexity Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Complexity_Analysis): Analysing the number of image parts after clustering and getting the complexity ratio
- [Image Content Analysis](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis): Getting fuzzy-labels for images with a resnet trained for the COCO-STUFF data set and then gathering statistics on word frequency in labels, including Google Cloud Vision  labels
	- [Cloud Vision Labeling](https://github.com/PonceLab/as-simple-as-possible/tree/main/Image_Content_Analysis/Google_Cloud_Vision.mlx): Getting labels from Google Cloud Vision API. 

### Update 07/16/2021
Work was completed on 07/06/2021 the remaining work is to:
* Produce markdown readmes or html for each live script so that users can see the work on the github page without having to download the livescripts first.
* Update this main readme to provide guidance for downloading and running
* complete some nice-to-haves for the ensembles and image content live scripts
* test on an clean computer with no dependencies pre-installed


### ~~Assignments for tasks within the PonceLab group
Work date: 07/06/2021~~

~~Guidelines: Everyone make a fork or just download the structure work on your computer then James will do the merge. Be mindful of what should probably go in the "utils" and "data" folder and do that. Keep the analysis folders limited to the livescript + any scripts or functions that absolutely could not fit in the livescript, and that no one else is likely to use in their folders. Thus, ranksum2 would go in utils, not in a livescript. All dependencies should be in either the analysis folder or the utils function unless they are owned by someone else (e.g. a FEX file, or the resnet repo etc.). For dependencies that are not able to be included, please directly list them at the top of the livescript.

~~CRP: 
Fill out the complexity code folder, and the ensembles folder

~~JKJ:
Fill out the Image Statistic Analysis folder and the COCO-stuff part of the image content analysis folder

~~WB:
Fill out the Google cloud vision part of the image content analysis folder (+ something about troubleshooting Anaconda and Matlab?)

~~After the work date:
James will do the merges, any cleaning that is needed and make sure everything imports like it should. He will put the dependents in a requirements.txt file and/or a script that can be run, or quality advice given to setting up MATLAB to work with python. 

### useful links

https://blogs.mathworks.com/pick/2021/01/08/convert-your-live-scripts-to-markdown-file/

https://www.markdownguide.org/cheat-sheet/
