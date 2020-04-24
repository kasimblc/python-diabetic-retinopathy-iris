from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(ResNet50(include_top=False,pooling="avg",weights="imagenet",input_shape=(128,128,3)))

model.add(Dense(5,activation="softmax"))


model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=["accuracy"])

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

dataGenerator=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.1)

trainGenerator=dataGenerator.flow_from_directory("D:\\mocococo\\Phyton_diabetic-retinopathy_iris\\iris",target_size=(128,128),
                                                 batch_size=8,class_mode="categorical",
                                                 subset="training")


valGenerator=dataGenerator.flow_from_directory("D:\\mocococo\\Phyton_diabetic-retinopathy_iris\\iris",target_size=(128,128),
                                                 batch_size=8,class_mode="categorical",
                                                 subset="validation")

fitHistory=model.fit_generator(trainGenerator,
                               steps_per_epoch=trainGenerator.samples//8,
                               epochs=45,
                               validation_data=valGenerator,
                               validation_steps=valGenerator.samples//8)

model.save_weights("diabetic-retinopathy-iris_weights-2.h5")
model.save("diabetic-retinopathy-Model-2.h5")




#modelLoad=load_model("diabetic-retinopathy-Model.h5")
#resimOkuma=dataGenerator.flow_from_directory("D:\\mocococo\\Phyton_diabetic-retinopathy_iris\\Predict",target_size=(128,128),
#                                                 batch_size=1,class_mode="categorical")
#tahmin=model.predict_generator(resimOkuma)