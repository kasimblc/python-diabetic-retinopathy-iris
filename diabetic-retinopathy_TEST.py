# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:13:52 2019

@author: sumak
"""
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model

modelLoad=load_model("C:\\Users\\sumak\\Desktop\\input\\diabetic-retinopathy-Model-2.h5")


resimOkuma=dataGenerator.flow_from_directory("D:\\mocococo\\Phyton_diabetic-retinopathy_iris\\Predict",target_size=(128,128),
                                                batch_size=1,class_mode="categorical")

tahmin=model.predict_generator(resimOkuma)

print(tahmin)
