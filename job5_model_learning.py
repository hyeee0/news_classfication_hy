import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.decomposition import PCA

X_train, X_test, Y_train, Y_test = np.load(
    'models/news_data_max_20_wordsize_11922.npy', allow_pickle=True
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(11922, 300, input_length=20))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1)) #1이면 달라지는게 없다
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128,
                     epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_calssfication_model_{}.h5'.format(
    np.round(fit_hist.history['val_accuracy'][-1], 3)))

plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


