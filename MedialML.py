import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import sklearn
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Conv3D, MaxPooling3D,GlobalAveragePooling3D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.decomposition import PCA

st.title("Medical dataset prediction using ML")
st.write("""
#Explore different ML algorithms
""")
datasets = st.sidebar.selectbox("Select Datasets",("Heart Disease","Diabetes","Breast Cancer","Covid-19","BP"))
st.write(datasets)


if datasets =="Heart Disease" or datasets=="Diabetes" or datasets =="Breast Cancer" or datasets=="BP":
    classifiers = st.sidebar.selectbox("Select Classifier",("KNN","SVC","Decision Tree","Random Forest"))
    st.write(classifiers)
    def load_datasets_csv(datasets):
        if datasets == "Heart Disease":
            data = pd.read_csv("C:/Users/khkir/Downloads/heart.csv")
        elif datasets == "Diabetes":
            data = pd.read_csv("C:/Users/khkir/Downloads/Diabetes.csv")
        elif datasets == "BP":
            data = pd.read_csv("C:/Users/khkir/Downloads/BP.csv")
            data["BP"] = data["Blood_Pressure_Abnormality"]
            data = data.drop("Blood_Pressure_Abnormality",axis=1)
        else:
            data = pd.read_csv("C:/Users/khkir/Downloads/Breast Cancer.csv")
            data["cancer"] = data["diagnosis"]
            data = data.drop(["diagnosis","Unnamed: 32"],axis=1)
        return data
    data = load_datasets_csv(datasets)
    st.write("Top 2 rows of the dataset ",data.head(2))
    st.write("Shape of the Data ",data.shape)
    st.write("Number of classes ", len(np.unique(data.iloc[:,-1:])))
    data = data.dropna() # If any null values are present in dataset it will be drop
    st.write("Describe the dataset ",data.describe())
    n_f = data.select_dtypes(include=[np.number]).columns
    c_f = data.select_dtypes(include=[np.object]).columns
    st.write("List of Numerical columns in data ",n_f)
    st.write("List of Categorial Columns in data ",c_f)
    data = pd.get_dummies(data)
    X = data.drop(data.iloc[:,-1:],axis=1)
    y = data.iloc[:,-1:]

    def add_parameters_csv(clf_name):
        p = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K",1,30)
            p["K"] = K
        elif clf_name == "SVC":
            C = st.sidebar.slider("C",0.01,15.0)
            p["C"] = C
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth",2,15)
            n_estimators = st.sidebar.slider("n_estimators",1,100)
            p["max_depth"] = max_depth
            p["n_estimators"] = n_estimators
        else:
            min_samples_split = st.sidebar.slider("min_samples_split",2,5)
            p["min_samples_split"] = min_samples_split
        return p
    p = add_parameters_csv(classifiers)

    def get_Classifier_csv(clf_name,p):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=p["K"])
        elif clf_name == "SVC":
            clf = SVC(C=p["C"])
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=p["n_estimators"],max_depth=p["max_depth"],random_state=1200)
        else:
            clf = DecisionTreeClassifier(min_samples_split=p["min_samples_split"])
        return clf
    clf = get_Classifier_csv(classifiers,p)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1200)
    clf.fit(X_train,y_train)
    y_pred_test = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred_test)
    st.write(f"classifier Used={classifiers}")
    st.write(f"accuracy score={acc}")
    
else:
    st.write("Using CNN algorithm")
    select_optimizer = st.sidebar.selectbox("Select an optimizer",("Adam","SGD"))
    st.write(select_optimizer)
    def load_datasets_images(datasets):
        if datasets == "Covid-19":
            path = "C:/Users/khkir/Downloads/archive/COVID-19_Radiography_Dataset"
        return path
    
    path = load_datasets_images(datasets)
    classes=["COVID",  "Normal"]
    num_classes = len(classes)
    st.write("Number of classes ",num_classes)
    batch_size=32
    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest", rescale=1./255, validation_split=0.3)
    train_gen = datagen.flow_from_directory(directory=path, 
                                              target_size=(299, 299),
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True, classes=classes,
                                              batch_size=batch_size, 
                                              color_mode="grayscale")
    #load the images to test
    test_gen = datagen.flow_from_directory(directory=path, 
                                              target_size=(299, 299),
                                              class_mode='categorical',
                                              subset='validation',
                                              shuffle=True, classes=classes,
                                              batch_size=batch_size, 
                                              color_mode="grayscale")
    def CNN():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(num_classes, activation='sigmoid'))
        return model
    
    model = CNN()
    def optimizer_selection(select_optimizer):
        if select_optimizer =="SGD":
            opt = SGD(lr=0.01, momentum=0.9)
        else:
            opt = tf.optimizers.Adam(learning_rate=0.1)
        return opt
    opt = optimizer_selection(select_optimizer)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    epoch=50
    history = model.fit(train_gen, validation_data=test_gen, epochs=epoch)
    scores = model.evaluate(test_gen)
    st.write("Scores ",scores)
    st.write('Model accuracy ',scores[1])




