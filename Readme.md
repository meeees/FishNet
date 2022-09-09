# FishNet
## `fish_id.py` and `fish_pred.py`
CNNs to identify types of fish. The saved model was trained to 93% accuracy on training data and 83% accuracy on testing data. It suffered from overfitting through most of the process, so after playing around with it for a few days I decided this was good enough.

I used this dataset for my training, so naturally it can only identify these types of fish: https://www.kaggle.com/datasets/khaledelsayedibrahim/fishclassifyfinal

<hr/>

## `make_randoom_fish.py` and `view_fish.py`

Generative Adversarial Network to try to generate fish. After training for 30,000 generations, I wrote a visualizer to play around with the 20 different input variables available to the model. It's not very good at making fish, even after I limited it to only oscars. Has the colors right and general shape sometimes.

I also removed most of the images with white backgrounds from the training set. I imagine more consistent training data would produce more consistent fish.

<hr/>

## Notes
The identifier seems to perform better when the fish is larger in the image, or maybe makes generalizations based on relative size of the fish. A better approach might be 2 models, one to locate the fish and restrict the input image to only suspected fish, then the model to evaluate the fish.


![image](https://user-images.githubusercontent.com/1458933/183224234-b4290e66-5cd2-4fed-8fa8-74803f45d494.png)
<br/>
<sub>**predict1.1.jpg is a scaled/cropped image of predict.jpg**</sub>
<hr/>


Also, plugging the generated image into the id model didn't produce an oscar. It really had no idea what it was looking at.

![image](https://user-images.githubusercontent.com/1458933/189251806-f8e2a84c-84fd-4a57-881d-9b4300656c41.png)
