# Learning to Find Anthropometric Landmarks
There are four directories in this repo: **/data**, **/preprocess**, **/learn**, and **/vis**. 
## data
There are two subdirectories, each has the following structure:
- ply: raw mesh data from CAESAR or FAUST
- lnd: landmark information of mesh
- processed: processed feature for learning

For demostration purpose, we only included 2 ply training shapes and 6 ply test shapes in this repo, this sample data can be found at https://drive.google.com/drive/my-drive. To download the full preprocessed training data set of FAUST, go to https://drive.google.com/file/d/1QT-cGkBIZ2elWU_GJ_nlwNhUh3TgeHMj/view?usp=sharing (~5G). Unzip **data.zip** and replace the **data** directory.
## preprocess
Go to directory **/preprocess** and run script **/preprocess_dataset.m**. This will preprocess all raw .ply data in directory **/data/ply** and compute SHOT descriptors, saving the processed data into **/data/processed**
## learn
To train models with processed data, at the root directory:
```
cd learn/learning_to_find_landmarks
python lnd_locating_ith.py [LND] [FLAG]
LND - the index of the target landmark from 0 to 72
FLAG - 1 indicates training mode; 0 indicates testing with pretrained model for target landmark LND
Example: "python lnd_locating_ith.py 8 1" will train for the landmark indexed 8.
```
After training, the trained model will be saved in **/model/save**

After testing, a point-wise potential value for testing mesh is saved in **/vis/res**, used for visualizing the result.

**Note**: we provided pretrained models for landmarks indexed 0, 8, 16, 32, 40, and 64.
## vis
To visualize predicted localization, run script **/vis/view_potential.m**
## Acknowledgement
We gratefully acknowledge the third party software and data used in this paper.

This software uses DiffusionNet[[1]](#1) for feature refinement. The source code is included here for convenience of review. The original code released by the authors may be found at https://github.com/nmwsharp/diffusion-net

The preprocess code is inherited from Deep Shells[[2]](#2), and the original code can be found at https://github.com/marvin-eisenberger/deep-shells

The dataset constains human scans from MPI FAUST[[3]](#3) Dataset https://faust-leaderboard.is.tuebingen.mpg.de

## References
<a id="1">[1]</a> N. Sharp, S. Attaiki, K. Crane, and M. Ovsjanikov. Diffusionnet: Discretization agnostic learning on surfaces. ACM Trans. Graph., 41(3), mar 2022. ISSN 0730-0301. doi: 10.1145/3507905. URL https://doi.org/10.1145/3507905.

<a id="2">[2]</a> M. Eisenberger, A. Toker, L. Leal-Taixé, and D. Cremers. Deep shells: Unsupervised shape correspondence with optimal transport. Advances in Neural Information Processing Systems, 34, 2020.

<a id="3">[3]</a> F. Bogo, J. Romero, M. Loper, and M. J. Black. FAUST: Dataset and evaluation for 3D mesh registration. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3794–3801, 2014.
