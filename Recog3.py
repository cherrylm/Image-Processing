import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras import datasets,models
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
#导入数据
data = pd.read_csv('train.csv')
y = data['label']
X = data.drop('label',axis=1)

#把train.csv的数据按照9:1的比例分为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=11)

x_train = x_train.values
y_train = y_train.values
x_val = x_val.values
y_val= y_val.values

#转化输入的格式并归一化
x_train = x_train.reshape((37800, 28, 28, 1)).astype('float32')
x_val = x_val.reshape((4200, 28, 28, 1)).astype('float32')
x_train, x_val = x_train / 255.0, x_val / 255.0

#导入Keras的MNIST数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

x_train = np.concatenate((train_images, x_train), axis=0)#训练集结合一起
x_val = np.concatenate((test_images, x_val), axis=0)#验证集放一起
y_train = np.concatenate((train_labels, y_train), axis=0)
y_val = np.concatenate((test_labels, y_val), axis=0)

#建立卷积神经网络的模型
input_shape = (28, 28, 1)
model = models.Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = (3,3),activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()

epochs_range = 50
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#训练模型
history = model.fit(x_train, y_train,batch_size=256,epochs=epochs_range,validation_data=(x_val, y_val) )
#评估模型
val_loss1, val_acc1 = model.evaluate(x_val, y_val,verbose=2)
print("Test Accuracy: ", val_acc1)
print("Test loss:", val_loss1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
e_range = range(epochs_range)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(e_range, acc, label='Training Accuracy')
plt.plot(e_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylim([0.98, 1])

plt.subplot(1, 2, 2)
plt.plot(e_range, loss, label='Training Loss')
plt.plot(e_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Loss')
plt.ylim([0.00, 0.05])
plt.show()
#保存模型
model.save('trained_cnn_model_20201224.model')
#对测试集进行数据预处理之后进行预测并保存输出结果
df_t = pd.read_csv('test.csv')
X_t = df_t.values
X_t = X_t.reshape((28000, 28, 28, 1)).astype('float32')
X_t = X_t/255.0
predictions = model.predict_classes(X_t, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission21_predictions_cnn_20201224.csv", index=False, header=True)