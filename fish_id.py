import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fish_id_plots import make_plot

data_dir = pathlib.Path('fish_data')
img_width = 200
img_height = 100

def make_model(num_classes) :
    reg = keras.regularizers.L1L2(l1=0, l2=0.000001)

    dropout = 0.2555
    rand = 0.15
    model = keras.Sequential([
        layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
        layers.RandomRotation(rand),
        layers.RandomZoom(rand),
        layers.RandomTranslation(rand, rand),
        layers.RandomContrast(rand),
        # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(64, 5, strides=(2,2), padding='same', activation='relu', kernel_regularizer=None),
        # layers.Dropout(dropout),
        layers.Conv2D(64, 3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=None),
        layers.SpatialDropout2D(dropout / 16),
        # layers.Dropout(dropout),
        layers.Conv2D(128, 3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=reg),
        # layers.SpatialDropout2D(dropout),
        layers.Conv2D(256, 3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=reg),
        layers.SpatialDropout2D(dropout),
        layers.Flatten(),
        # kernel_regularizer=keras.regularizers.L1L2(l1=0.00001,l2=0.0001)
        layers.Dense(num_classes * 4, activation='relu', kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(num_classes, kernel_regularizer=None),
        layers.Softmax()
    ])
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.002, name='crossentropy')
    # loss = tf.keras.losses.CategoricalCrossentropy(name='crossentropy')
    optimizer = tf.keras.optimizers.Adamax()
    metrics=[loss, 'accuracy']


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == '__main__' :

    batch_size = 64
    seed = 1338

    val_split = 0.1

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        seed = seed,
        subset = 'training',
        validation_split = val_split,
        image_size = (img_height, img_width),
        batch_size = batch_size,
        label_mode='categorical')

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        seed = seed,
        subset = 'validation',
        validation_split = val_split,
        image_size = (img_height, img_width),
        batch_size = batch_size,
        label_mode='categorical')

    class_names = (train_ds.class_names)
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    model = make_model(num_classes)


    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_crossentropy', patience=75)
    model.summary()

    # model.load_weights('trained_v5_50')

    epochs = 4000

    history = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = [earlyStop] )

    model.save('trained_v6')

    make_plot(history)

