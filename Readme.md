# FishNet
## Overview
CNNs to identify types of fish. The saved model was trained to 93% accuracy on training data and 83% accuracy on testing data. It suffered from overfitting through most of the process, so after playing around with it for a few days I decided this was good enough.

I used this dataset for my training, so naturally it can only identify these types of fish: https://www.kaggle.com/datasets/khaledelsayedibrahim/fishclassifyfinal

## Notes
It seems to perform better when the fish is larger in the image, or maybe makes generalizations based on relative size of the fish. A better approach might be 2 models, one to locate the fish and restrict the input image to only suspected fish, then the model to evaluate the fish.

(predict.jpg results vs predict1.1.jpg)

![image](https://user-images.githubusercontent.com/1458933/183224234-b4290e66-5cd2-4fed-8fa8-74803f45d494.png)

## Future
Longer term goal is to use this model to train other models that can create images that are supposedly fish of a specific species.
