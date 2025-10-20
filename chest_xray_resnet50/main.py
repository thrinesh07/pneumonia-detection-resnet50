train_dir=r"T:\datasets\X-rays_dataset\chest_xray\train"
val_dir=r"T:\datasets\X-rays_dataset\chest_xray\val"
test_dir=r"T:\datasets\X-rays_dataset\chest_xray\test"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    rescale=1./225
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_data=train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

val_data=val_datagen.flow_from_directory(
    val_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

test_data=test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

from keras.applications import ResNet50
base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(128,128,3))
base_model.trainable=False

from keras.models import Sequential
from keras.layers import Dense,Dropout,GlobalAveragePooling2D

model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256,activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping
eraly_stop=EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)


store=model.fit(train_data,validation_data=val_data,epochs=100,callbacks=[eraly_stop],verbose=1)

loss,acc=model.evaluate(test_data)
print(f'ACC : {acc:.2f}')


import matplotlib.pyplot as plt

plt.plot(store.history['accuracy'],label='Train ACC')
plt.plot(store.history['val_accuracy'],label='val Acc')
plt.title("ResNet50")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


X_val,y_val=next(val_data)
pred=model.predict(X_val)
pred_class=(pred>0.5).astype("int32")

for i in range(5):
    plt.imshow(X_val[i])
    label="NORMAL" if pred_class[i]  == 0 else "PNEUMONIA"
    plt.title(label)
    plt.axis('on')
    plt.show()