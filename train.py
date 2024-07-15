import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

(X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()

model.add(Conv2D(16,(3,3), activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

adam = tensorflow.keras.optimizers.Adam()

model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

model.save("digit_recognition.h5")
