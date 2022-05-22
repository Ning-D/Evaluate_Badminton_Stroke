# Deep reinforcement learning in a racket sport for player evaluation with technical and tactical contexts 

## Author 
Ning Ding

## Reference:
Ning Ding, Kazuya Takeda, Keisuke Fujii, Deep reinforcement learning in a racket sport for player evaluation with technical and tactical contexts, IEEE Access, accepted.

## Requirements:

* Python 2.7

* Numpy 1.16.5
* Tensorflow 1.14.0
* Scipy
* Matplotlib
* scikit-learn 

## External Dependencies:

* [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
* [TrackNet](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet)

## Data:

* Video data can be downloaded from [Youtube](https://www.youtube.com/user/bwf). To run our model directly, you can also download the preprocessd data from [Dropbox](https://www.dropbox.com/sh/vxxgnnsn8ze0usl/AAD7ew6tQsgymsacemU0y6uSa?dl=0)


## Usage 

## Training:
1. Modify the save_mother_dir in configuration.py as your save directory
2. Cd into your save_mother_dir, make two directories ./models/hybrid_sl_saved_NN/ and ./models/hybrid_sl_log_NN/
3. Download the preprocessd data.
4. Run ```python Train.py```  
5. The trained model will be saved in the file (e.g. ./saved_models_gammaXX_hdXX_iterXX_lrXX)

## Evaluation:
1. Run ```python Evaluate.py```  
2. To obtain the result of action value in a badminton rally. Run ```python plot.py --iter_number xx```
 
## Acknowledgements:
For this project, we relied on research codes from:
* [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
* [TrackNet](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet)
* [DRL-ice-hockey](https://github.com/Guiliang/DRL-ice-hockey)
