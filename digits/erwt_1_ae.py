from sklearn import linear_model
from sklearn.model_selection import cross_val_score, GridSearchCV 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

digits = scipy.io.loadmat("digits.mat")
X = digits["learn"][0,0][0]
X = np.swapaxes(X,0,1)
X= X/16.

print X

y = digits["learn"][0,0][1]
y =np.squeeze(y)

encoding_dim = 32

#import dataset
x_train, x_test = train_test_split(
    X, test_size=0.10, random_state=42)

input_img = Input(shape=(64,))
encoded = Dense(60, activation='relu')(input_img)
encoded = Dense(55, activation='relu')(encoded)
encoded = Dense(50, activation='relu')(encoded)

decoded = Dense(55, activation='relu')(encoded)
decoded = Dense(60, activation='relu')(decoded)
decoded = Dense(64, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoded)
encoded_input_1 = Input(shape=(50,))
encoded_input_2 = Input(shape=(55,))
encoded_input_3 = Input(shape=(60,))

decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch= 100,
                batch_size=40,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder_1.predict(encoded_imgs)

