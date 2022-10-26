

## For Object Features 

``` bash
from efficientnet_pytorch import EfficientNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3


modelVgg = VGG16(weights='imagenet', include_top=False) # include_top=False with five output of second last layer before classification
modelInceptionV3=InceptionV3(weights='imagenet', include_top=False)
modelEff= EfficientNet.from_pretrained('efficientnet-b0')

vgg16_feature = modelVgg.predict(frmRz224)
inceptionv3_feature = modelInceptionV3.predict(frmRz299)

```
            
for more details see "extractFeatures_Inception_vgg16_efficientnet.py"
or visit
https://github.com/VideoAnalysis/EDUVSUM/tree/master/src



## For Kinetic Features(FLOW & RGB)

1- Download checkpoints "https://github.com/deepmind/kinetics-i3d/tree/master/data/checkpoints/rgb_imagenet". 

2-Extract video frames (like: fps=2).
``` bash
ffmpeg -i [video_input_path] -r 2 [video_save_dir]/%d.jpg
```

3- Run feature extractor.
```
python feature_extractor_frm.py
```
4- Generate feature h5 file.
```
python get_i3d_h5.py
```

Reference:
https://github.com/JaywongWang/I3D-Feature-Extractor
https://github.com/Finspire13/pytorch-i3d-feature-extraction

Note:
if you haev confusion in loading checkpoints, go through the link
https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
