import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pylab as py
import cv2
import scipy.misc
import warnings
import os
import keras
import pydot
import pydot_ng
import time
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import tarfile
import wget
import zipfile
import shutil
import statistics
import random
import h5py

from astropy.io import fits

#####3
from keras import applications
from keras_applications import vgg16
#####3

from keras import models

####
from keras.models import Sequential
from keras import layers
####

from keras.layers import Input

#########
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
#######

from pathlib import Path
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import plot_model
from IPython.display import Image, display
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import KFold  #para cross-validation
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.applications.resnet50 import ResNet50
from sklearn import metrics
from sklearn.metrics import confusion_matrix

########################################################
from utilities import fileremover, filemover, load_data_kfold, get_model, get_callbacks, HighestInteger, ROCCurveCalculate, data_downloader, TestSamplesBalancer
########################################################

warnings.filterwarnings("ignore")
tf.config.experimental.set_visible_devices([], 'GPU')

print("\n\n ## Tensorflow version:")
print(tf.__version__)

print("\n\n ## Is GPU available?")
print(tf.test.is_gpu_available())

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

###############################################PARAMETERS

num_epochs = 50
batch_size = 4
nk = 10     #NÚMERO DE FOLDS
version = 4        #VERSÃO PRA COLOCAR NAS PASTAS
PC = 1.0      #PORCENTAGEM DO DATASET DE TESTE A SER USADO
num_classes = 2
classes = ['lens', 'not-lens']
width = 0.35

########################################################

print("\n ** Verifying data...")

#data_downloader()

print("\n ** Reading data from y_data20000fits.csv...")

#LOAD Y_DATA
PATH = os.getcwd()
var = PATH + "/" + "lensdata/"
y_batch = os.listdir(var)

#HERE WE'RE GOING TO START THE MULTIPLE PROGRAMS
study = [300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 550, 600, 650, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 5000, 10000, 18000]
#study = [90, 80, 70, 60, 50, 40, 30, 20, 10]
print(study)

begin = time.perf_counter()

for u in range(0,len(study)):
    try:
        TR = study[u]

        print('\n\n\n ** Beginning new cicle with number of samples %s ! **************************************************************************************************' % TR)

    ####################################3

        print('\n\n ** Cleaning up previous files and folders...')

        fileremover(TR, nk, version)       
        ######################################################
       
        print("\n ** Starting data preprocessing...")

        labels = pd.read_csv(var + 'y_data20000fits.csv',delimiter=',', header=None)
        y_data = np.array(labels, np.uint8)
        y_size = len(y_data)
        y_data.shape

        x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')
        Ni_channels = 0 #first channel
        N_channels = 3 #number of channels

        x_data = x_datasaved['data']
        x_size = len(x_data)
        x_data = x_data[:,:,:,Ni_channels:Ni_channels + N_channels]

        print(" ** Randomizing y_data and x_data...")
        ind = np.arange(y_data.shape[0])
        np.random.shuffle(ind)
        y_data = y_data[ind]
        x_data = x_data[ind]

        print(" ** y_data has shape: ", y_data.shape)
        print(" ** Total dataset size: ", y_size, "objects.")

        print(' ** Balancing number of samples on each class for train+val sets with TR samples...')     

        y_data, x_data, y_test, x_test = TestSamplesBalancer(y_data, x_data, TR)

        y_size = len(y_data)
        y_tsize = len(y_test)
        x_size = len(x_data)

        print(" ** y_data arranged with format:")
        print(" ** y_test:   ", y_test.shape)
        print(" ** y_data:  ", y_data.shape)

        trainval_count = [np.count_nonzero(y_data == 1), np.count_nonzero(y_data == 0)]
        test_count = [np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)]

        #############DISTRIBUTION GRAPH#########
        plt.figure()
        fig, ax = plt.subplots()
        ax.bar(classes, test_count, width, label='Test')
        ax.bar(classes, trainval_count, width, bottom=test_count, label='Train+Val')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset distribution')
        ax.legend(loc='bottom right')
        fig.savefig("TrainTest_rate_TR_{}.png". format(TR))
        ########################################
 
        #PC = AMOUNT OF THE DATASET USED (< 1 FOR TESTING)
        y_test = y_test[0:int(y_size*PC),]
        x_test = x_test[0:int(y_size*PC),:,:,:]
        ##############    

        print("\n ** x_data splitted with format:")
        print(" ** x_test:   ", x_test.shape)
        print(" ** x_data:  ", x_data.shape)

        print("\n ** Converting data and list of indices into folds for cross-validation...")

        subset_size = int(y_size/nk)   #nk = number of folds
        print(y_size, subset_size)
  
        folds = load_data_kfold(nk, x_data, y_data)
        print("\n ** x_data splitted with shape: \n", x_data.shape, "\n ** y_data splitted with shape: \n", y_data.shape)

        print("\n ** Starting network training... \n")

        start = time.perf_counter()
        FPRall, TPRall, AUCall, acc0, loss0, val_acc0, val_loss0, lauc = ([] for i in range(8))
        y_test = to_categorical(y_test,num_classes=2)

        for j, (train_idx, val_idx) in enumerate(folds):
    
            print('\n ** Fold ',j)
            x_data_cv = x_data[train_idx]
            y_data_cv = y_data[train_idx]
            x_val_cv = x_data[val_idx]
            y_val_cv= y_data[val_idx]

            train_count = [np.count_nonzero(y_data_cv == 1), np.count_nonzero(y_data_cv == 0)]
            val_count = [np.count_nonzero(y_val_cv == 1), np.count_nonzero(y_val_cv == 0)]

            #############DISTRIBUTION GRAPH#########
            plt.figure()
            fig, ax = plt.subplots()
            ax.bar(classes, train_count, width, label='Train')
            ax.bar(classes, val_count, width, bottom=train_count, label='Validation')
            ax.set_ylabel('Number of Samples')
            ax.set_title('Data distribution on fold %s with %s samples)' % (j, TR))
            ax.legend(loc='bottom right')
            fig.savefig("TrainVal_rate_TR_{}_Fold_{}.png". format(TR, j))
            ########################################

            print("\n ** Converting vector classes to binary matrices...")
            y_data_cv = to_categorical(y_data_cv,num_classes=2)
            y_val_cv = to_categorical(y_val_cv,num_classes=2)

            print("\n ** Building ResNet model...")
    
            model = get_model(x_data, y_data, N_channels)
            model.summary()
            #model.save('ModelResnet50Lens.h5') # if save model
            print("\n ** Compiling model...")

            lr = 0.01
            sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss= 'binary_crossentropy' , optimizer='sgd' , metrics=[ 'accuracy' ])

            print("\n ** Plotting model and callbacks...")
            gen = ImageDataGenerator(rotation_range = 90)

            name_weights = "Train_model_weights_{epoch:02d}.h5"
            csv_name = "training_k.csv"
            callbacks = get_callbacks(name_weights = name_weights, patience_lr=10, name_csv = csv_name)
            generator = gen.flow(x_data_cv, y_data_cv, batch_size = batch_size)
    # change epochs
            history = model.fit_generator(
                          generator,
                          steps_per_epoch=len(x_data_cv)/batch_size,
                          epochs=num_epochs,
                          verbose=1,
                          validation_data = (x_val_cv, y_val_cv),
                          validation_steps = len(x_val_cv)/batch_size,
                          callbacks = callbacks)
      
      
            print("\n ** Training completed.")
            print("\n ** Plotting training & validation accuracy values.")  
    
            accu = history.history['accuracy']
            c = HighestInteger(accu, num_epochs)

        ###### Plot training & validation accuracy values
            plt.figure()
            plt.xlim([0,num_epochs])
            plt.ylim([0,c])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy' )
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.show()
            plt.savefig("AccxEpoch_{}_Fold_{}.png". format(TR, j))

            loss = history.history['loss']
            c = HighestInteger(loss, num_epochs)

            print("\n ** Plotting training & validation loss values.")
        ###### Plot training & validation loss values
            plt.figure()
            plt.xlim([0,num_epochs])
            plt.ylim([0,c])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.show()
            plt.savefig("LossxEpoch_{}_Fold_{}.png". format(TR, j))

            print("\n ** Model evaluation stage.")
        #print(model.evaluate(x_test, y_test)) 
        ######## calculate roc curve
    #    fpr, tpr, thresholds = roc_curve(y_test[:,1], probsp)
            tpr, fpr, auc, auc2, thres = ROCCurveCalculate(y_test, x_test, model)
            lauc = np.append(lauc, auc)
            AUCall.append(auc2)
            FPRall.append(fpr)
            TPRall.append(tpr)

            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--') # k = color black
            plt.plot(FPRall[j], TPRall[j], label="fold" + str(j) + "& AUC: %.3f" % auc, color='C'+str(j), linewidth=3) # for color 'C'+str(j), for j[0 9]
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
            plt.title('ROC')
            plt.xlabel('false positive rate', fontsize=14)
            plt.ylabel('true positive rate', fontsize=14)
    
            plt.show()
            plt.savefig("ROCLensDetectNet_{}_Fold_{}.png". format(TR, j))

            acc0 = np.append(acc0,history.history['accuracy'])
            val_acc0 = np.append(val_acc0,history.history['val_accuracy'])
            loss0 = np.append(loss0,history.history['loss'])
            val_loss0 = np.append(val_loss0,history.history['val_loss'])

            scores = model.evaluate(x_test, y_test, verbose=0)
            print(" ** Large CNN Error: %.2f%%" % (100-scores[1]*100))


        print('\n ** Training and evaluation complete.')
        elapsed = time.perf_counter() - start
        print(' ** Elapsed %.3f seconds.' % elapsed)

        print('\n ** Generating ultimate ROC graph...')

        medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--') # k = color black

        mauc = np.percentile(lauc, 50.0)
        mAUCall = np.percentile(AUCall, 50.0)
        plt.title('ROC')
        plt.xlabel('false positive rate', fontsize=14)
        plt.ylabel('true positive rate', fontsize=14)

        for num in range(0,int(thres),1):
            lis = [item[num] for item in TPRall]
            los = [item[num] for item in FPRall]
            
            medians_x.append(np.percentile(los, 50.0))
            medians_y.append(np.percentile(lis, 50.0))
            lowlim.append(np.percentile(lis, 15.87))
            highlim.append(np.percentile(lis, 84.13))
        
        lowauc = metrics.auc(medians_x, lowlim)
        highauc = metrics.auc(medians_x, highlim)
        print('\n\n\n ** IS THIS CORRECT?')
        print(lowauc, mauc, highauc)
        print(lowauc, mAUCall, highauc)

        plt.plot(medians_x, medians_y, 'b', label = 'AUC: %s' % mauc, linewidth=3)  
        plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha='0.3', interpolate=True)
        plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha='0.3', interpolate=True)
        plt.legend(loc='lower right', ncol=1, mode="expand")

        plt.savefig("ROCLensDetectNet_Full_%s.png" % TR)

######################################

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--') # k = color black

        plt.title('ROC')
        plt.xlabel('false positive rate', fontsize=14)
        plt.ylabel('true positive rate', fontsize=14)
        low = np.percentile(lowlim, 50.0)
        high = np.percentile(highlim, 50.0)

        plt.plot(medians_x, medians_y, 'b', label = 'AUC: %s' % mauc, linewidth=3)  
        plt.fill_between(medians_x, medians_y, low, color='blue', alpha='0.3', interpolate=True)
        plt.fill_between(medians_x, high, medians_y, color='blue', alpha='0.3', interpolate=True)
        plt.legend(loc='lower right', ncol=1, mode="expand")

        plt.savefig("ROCLensDetectNet_Test_%s.png" % TR)

        source = '/home/kayque/LENSLOAD/'
        script = '/home/kayque/LENSLOAD/SCRIPTS/'
    
        filemover(source, script, TR, version, nk)    

    except AssertionError as error:
        print(error)
    #except:
    #    pass

print(' ** Cleaning up residual files...')

fileremover(TR, nk)

timee = time.perf_counter() - begin
print('\n ** Mission accomplished with %s seconds.' % time)
print("\n ** FINISHED! ************************")
