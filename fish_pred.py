import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import PIL

from fish_id import make_model, img_width, img_height

from keras import backend as K

import pathlib

def load_image(path) :
    img = keras.utils.load_img(path, target_size=(img_height, img_width))
    # img.show()
    img = keras.preprocessing.image.img_to_array(img)
    return np.array([img])

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def do_the_thing(model, path) :
    res = model.predict(load_image(path))

    guess_1 = np.argmax(res)
    res2 = np.delete(res, guess_1)
    guess_2 = np.argmax(res2)
    if(guess_2 >= guess_1) :
        guess_2 += 1

    print(f'Guesses for {path}')
    print(f'{classes[guess_1]} : {res[0][guess_1] * 100}%')
    print(f'{classes[guess_2]} : {res[0][guess_2] * 100}%')
    # print(res)

def output_layers(model, inp, out_path) :                 
    model_inp = model.input                        # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([model_inp], [out]) for out in outputs]    # evaluation functions



    # Testing
    layer_outs = [func([inp]) for func in functors]
    convdepth = 5
    for i in range(convdepth, convdepth+5) :
        if i == convdepth + 2 :
            # dropout layer
            continue
        out = tf.squeeze(layer_outs[i])
        filters = len(out[0][0])
        print(filters)
        path = pathlib.Path(f'./{out_path}/{model.layers[i].name}')
        path.mkdir(parents=True, exist_ok=True)
        for j in range(filters) :
            slice = out[:, :, j]
            img = tensor_to_image(slice)
            p = path / f'{j}.png'
            img.save(p)



if __name__ == '__main__':
    classes = ['Black-winged hatchetfish', 'Bristlenose_catfish', 'Electric blue cichild', 'Electric_fish', 'Flowerhorn cichlid',
     'Gold', 'Guppy', 'Oscar Fish', 'Paradise Fish', 'Powder_blue_fish', 'Red_tail_black_shark', 'Sea Horse Fish',
      'Thalassoma_bifasciatum', 'Yellow Tang', 'bala', 'blood-red jewel cichild', 'blue zebra angelfish', 'bolivian ram', 'botia striata fish', 
'celestial eye goldfish', 'clown', 'clown loach', 'damsel fish', 'figure eight puffer', 'lion', 'neon goby', 'neon tetra', 'symphysodon discus', 'zebra danio fish']
    
    model = make_model(29)

    model.load_weights('best_so_far')

    # model.summary()

    # do_the_thing(model, 'hatchet.jpg')
    do_the_thing(model, 'tetra.png')
   #  output_layers(model, load_image('electricblue.jpg'), 'conv_output/electricblue.jpg')
    # do_the_thing(model, 'seahorse33.jpg')
    # do_the_thing(model, 'predict1.1.jpg')
    # do_the_thing(model, 'predict2.jpg')

    




