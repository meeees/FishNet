from audioop import cross
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import os
import time
import matplotlib.pyplot as plt

def rand_fish_model():
    i_w = 10
    i_h = 5
    model = tf.keras.Sequential()
    model.add(layers.Dense(i_h*i_w, use_bias=False, input_shape=(20,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(i_h * i_w * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((i_h, i_w, 256)))

    k = (5, 5)
    k2 = (3, 3)
    s = (2, 2)
    s2 = (1, 1)

    def double_layer() :
        model.add(layers.Conv2DTranspose(128, k, strides=s, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, k2, strides=s2, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
    # 20 x 10
    double_layer()

    # 40 x 20
    double_layer()
    
    # 80 x 40
    double_layer()

    # 160 x 80
    double_layer()

    model.add(layers.Conv2DTranspose(3, (3, 3), strides=s2, padding='same', use_bias=False))
    assert model.output_shape == (None, 80, 160, 3)
    
    model.summary()

    return model

def rand_fish_disc() :
    model = tf.keras.Sequential()
    s = (2, 2)
    k = (5, 5)
    model.add(layers.Conv2D(32, k, strides=s, padding='same', input_shape=[80, 160, 3]))
    model.add(layers.Conv2D(64, k, strides=s, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, k, strides=s, padding='same'))
    model.add(layers.Conv2D(128, k, strides=s, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())

    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(1))
    model.summary()
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(fake) :
    return cross_entropy(tf.ones_like(fake), fake)

def dis_loss(real, fake) :
    r_loss = cross_entropy(tf.ones_like(real), real)
    f_loss = cross_entropy(tf.zeros_like(fake), fake)
    return r_loss + f_loss


def show_tensor(t) :
    img = tf.keras.preprocessing.image.array_to_img(t[0])
    img.show()

offset = 13538
def gen_and_save(model, epoch, inp) :
    pred = model(inp, training=False)
    fig = plt.figure(figsize=(8, 8))

    for i in range(pred.shape[0]) :
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.clip_by_value(pred[i, :, :, :], 0, 1))
        plt.axis('off')
    try :
        plt.savefig(f'random_out/oscar_{epoch + offset}.png')
    except:
        try :
            plt.savefig(f'random_out/oscar_{epoch + offset}.png')
        except:
            print (f'skipped oscar_{epoch + offset}.png after failing twice')

    plt.close(fig)
    # plt.show()




BUFFER_SIZE = 500
BATCH_SIZE = 128

if __name__ == '__main__' :
    generator = rand_fish_model()
    discriminator = rand_fish_disc()

    gen_o = tf.keras.optimizers.Adam(1e-4)
    dis_o = tf.keras.optimizers.Adam(1e-4)


    AUTOTUNE = tf.data.AUTOTUNE
    data_dir = pathlib.Path('oscar_no_whites')
    r_seed = 1338
    fish_data = keras.utils.image_dataset_from_directory(
        data_dir,
        seed = r_seed,
        image_size = (80, 160),
        batch_size = BATCH_SIZE,
        labels=None).cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)

    # print(fish_data.take(1).get_single_element())
    # show_tensor(fish_data.take(1).get_single_element())

    checkpoints = './random_fish_checkpoints'
    cp_prefix = os.path.join(checkpoints, 'ckpt')
    checkpoint = tf.train.Checkpoint(gen_o=gen_o,
     dis_o = dis_o,
     generator = generator,
     discriminator = discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoints))

    EPOCHS = 20000
    noise_dim = 20
    num_gen = 16
    
    seed = tf.random.uniform([num_gen, noise_dim], seed=r_seed)

    @tf.function
    def train_step(images):
        noise = tf.random.uniform([BATCH_SIZE, noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_i =  generator(noise, training=True)

            images /= tf.constant(256, dtype=tf.float32)
            real_o = discriminator(images, training=True)
            fake_o = discriminator(gen_i, training=True)

            g_loss = gen_loss(fake_o)
            d_loss = dis_loss(real_o, fake_o)

            gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
            disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

            gen_o.apply_gradients(zip(gen_grads, generator.trainable_variables))
            dis_o.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            return g_loss, d_loss
    
    def train(dataset, epochs):
        for epoch in range(epochs) :
            start = time.time()

            for img_b in dataset:
                g_loss, d_loss = train_step(img_b)
                
                gen_and_save(generator, epoch, seed)

            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix=cp_prefix)
            
            print(f'Epoch {epoch}: {time.time() - start}s, g_loss: {g_loss}, d_loss: {d_loss}')
    
    # train(fish_data, EPOCHS)

    # generator.save_weights('best_so_far/random_fish/gen')
    # discriminator.save_weights('best_so_far/random_fish/dis')

    