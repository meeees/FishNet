# FishNet
## `fish_id.py` and `fish_pred.py`
CNNs to identify types of fish. The saved model was trained to 93% accuracy on training data and 83% accuracy on testing data. It suffered from overfitting through most of the process, so after playing around with it for a few days I decided this was good enough.

I used this dataset for my training, so naturally it can only identify these types of fish: https://www.kaggle.com/datasets/khaledelsayedibrahim/fishclassifyfinal

<hr/>

## `make_random_fish.py` and `view_fish.py`

Generative Adversarial Network to try to generate fish. After training for 30,000 generations, I wrote a visualizer to play around with the 20 different input variables available to the model. It's not very good at making fish, even after I limited it to only oscars, but it is *way* better than it started (random noise). It has the colors right and general shape sometimes
(see fake_fish.png and fake_mess.png).

I also removed most of the images with white backgrounds from the training set. I imagine more consistent training data would produce more consistent fish.

<hr/>

## Notes
The identifier seems to perform better when the fish is larger in the image, or maybe makes generalizations based on relative size of the fish. A better approach might be 2 models, one to locate the fish and restrict the input image to only suspected fish, then the model to evaluate the fish.


![image](https://user-images.githubusercontent.com/1458933/183224234-b4290e66-5cd2-4fed-8fa8-74803f45d494.png)
<br/>
<sub>**predict1.1.jpg is a scaled/cropped image of predict.jpg**</sub>

Also, plugging the generated image into the id model didn't produce an oscar. It really had no idea what it was looking at.

![image](https://user-images.githubusercontent.com/1458933/189251806-f8e2a84c-84fd-4a57-881d-9b4300656c41.png)

<hr/>
Some images from training

Early (0-10k):<br/>
<img src=https://user-images.githubusercontent.com/1458933/189252190-bb044b75-09f8-44ce-8a88-308e222d56d5.png width=400 height=400/>
<img src=https://user-images.githubusercontent.com/1458933/189252792-39f6ac43-3290-4a8a-b677-b59d6fa52284.png width=400 height=400/>

Mid (10k-20k):<br/>
<img src=https://user-images.githubusercontent.com/1458933/189252248-f236bfee-1fb5-43a3-8501-7557fadb9ef1.png width=400 height=400/>
<img src=https://user-images.githubusercontent.com/1458933/189252702-28bdd778-4953-40e0-b92f-4ba6d256a714.png width=400 height=400/>


Late (20k+):<br/>
<img src=https://user-images.githubusercontent.com/1458933/189252910-71ac906b-5e94-4f9f-9aac-a41822233b34.png width=400 height=400/>
<img src=https://user-images.githubusercontent.com/1458933/189252105-458b231d-49fa-42d7-9673-014ec3308b60.png width=400 height=400/>

full video - https://www.youtube.com/watch?v=ghFB0SqnZlQ


