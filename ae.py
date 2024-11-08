import keras
import tensorflow as tf
from keras import layers
from keras.losses import mse
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # load data
    x_window_normal=np.load("datasets//windows_normal_flatten.npy")
    x_all=np.load("datasets//windows_attack_flatten.npy")
    # Declare the number of nodes for input layer, hidden layer, and latent layer (space)
    window_size=12
    input_size=51
    window_dim =window_size*input_size
    latent_dim = 100

    # Build the encoder part
    original_inputs = tf.keras.Input(shape=(window_dim,), name="encoder_input")
    #x = layers.Dense(window_dim, activation="relu")(original_inputs)
    x = layers.Dense(window_dim/2, activation="relu")(original_inputs)
    x = layers.Dense(window_dim/4, activation="relu")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    AE_encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

    # "Define decoder model:
    # z -> hidden layer -> output"
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(window_dim/4, activation="relu")(latent_inputs)
    x = layers.Dense(window_dim/2, activation="relu")(x)
    outputs = layers.Dense(window_dim, activation="sigmoid")(x)
    AE_decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

    # Build up the Auto-Encoder Model
    outputs = AE_decoder(z)
    ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="auto-encoder")
    ae.summary()

    # the input of encoder should match the output of the decoder
    reconstruction_loss = mse(original_inputs, outputs)

    # Add the loss to the model
    ae.add_loss(reconstruction_loss)
    ae.add_metric(reconstruction_loss, name='mse_loss', aggregation='mean')

    # Optimizer is Adam, learning rate is 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ae.compile(optimizer)
    # adaptive learning rate
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9, patience=20, min_lr=1e-20)]


    #Training
    start_time=time.time()
    history = ae.fit(x=x_window_normal[:int(len(x_window_normal)*0.7)], y=x_window_normal[:int(len(x_window_normal)*0.7)], epochs=30, batch_size=8000, callbacks=callbacks)
    print("Training time:", time.time() - start_time)
    # get anomaly score for each window and append it to a score list
    x_pred=ae(x_all).numpy()
    score_list = []
    for i in range(x_all.shape[0]):
        score = mse(x_all[i], x_pred[i])
        score_list.append(score.numpy())
    np.save("datasets//ae_score.npy", np.array(score_list))



    # plot losses
    plt.plot(history.history["loss"])
    #plt.plot(history.history["val_loss"])
    plt.suptitle('Training result')
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
