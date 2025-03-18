import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from math import sqrt

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def stochastinis_GD(dataset, validation, test):
    #random.seed(1)

    in_num = 9 
    out_num = 9 
    limit = sqrt(6 / (in_num + out_num))
    W = [random.uniform(-limit, limit) for _ in range(in_num)]
    bias = random.uniform(-limit, limit)

    y = dataset.pop('Class')
    y_val = validation.pop('Class')
    y_test = test.pop('Class')

    totalError = 999999
    epoch = 0
    totalErrorList = []             #visi lists skirti vizualizacijoms
    totalErrorListVal = []
    y_pred= []
    y_pred_val = []
    y_pred_test = []
    accuracies_train= []
    accuracies_validate = []

    minError = 0.02
    epochs = 300
    learning_rate = 0.5

    m = dataset.shape[0]
    m_val = validation.shape[0]
    m_test = test.shape[0]
    n = len(W)

    print("searching...")
    start = time.time()

    while (totalError > minError and epoch < epochs):

        totalError = 0

        for i in range(0, m): 

            a = (dataset.iloc[i] * W).sum() + bias #daugina su visais eilutes W elementais
            yi = sigmoid(a)
            t = y.iloc[i]
            y_pred.append(round(yi, 0))

            for k in range(0, n): 
                W[k] = W[k] - learning_rate*( yi - t )*yi*( 1 - yi )*dataset.iloc[i, k]
            bias = bias - learning_rate * (yi - t) * yi * (1 - yi)

            error = (t - yi)**2
            totalError = totalError + error

        #MOKYMOSI TIKSLUMAS
        acc = accuracy_score(y, y_pred)
        accuracies_train.append(acc)
        y_pred.clear()


        #VALIDAVIMAS!!!
        totalErrorVal=0
        for i in range(0, m_val): 

            a = (validation.iloc[i] * W).sum() + bias
            yi = sigmoid(a)
            
            y_pred_val.append(round(yi, 0))

            t = y_val.iloc[i]
            error = (t - yi)**2
            totalErrorVal = totalErrorVal + error

        #VALIDAVIMO TIKSLUMAS
        acc_val = accuracy_score(y_val, y_pred_val)
        accuracies_validate.append(acc_val)
        y_pred_val.clear()

        totalErrorListVal.append(totalErrorVal/m_val)
        totalError=totalError/m
        totalErrorList.append(totalError)
        epoch+=1


    end = time.time()
    print('> Neurono mokymosi laikas sekundemis: ', round((end - start),3))

    #testavimas
    totalError = 0
    for i in range(0, m_test): 

        a = (test.iloc[i] * W).sum() + bias #daugina su visais eilutes W elementais
        yi = sigmoid(a)
        y_pred_test.append(round(yi, 0))

        t = y_test.iloc[i]
        error = (t - yi)**2
        totalError = totalError + error
        
    totalError = totalError/m_test
    acc_test = accuracy_score(y_test, y_pred_test)

    
    print('> Testavimo paklaida:', round(totalError,2))
    print('> Testavimo tikslumas:', round(acc_test,2))

    epoch_numbers = list(range(1, epoch + 1))
    plt.plot(epoch_numbers, totalErrorList, color='red', label='Mokymasis')
    plt.plot(epoch_numbers, totalErrorListVal, color='blue', label='Validavimas')

    plt.title('Paklaida')
    #plt.xticks(np.arange(min(epoch_numbers), max(epoch_numbers)+1, 1))
    plt.legend()
    plt.show()
    plt.clf()


    plt.plot(epoch_numbers, accuracies_train, color='red', label='Mokymasis')
    plt.plot(epoch_numbers, accuracies_validate, color='blue', label='Validavimas')

    plt.title('Tikslumas')
    plt.ylim(0, 1)
    #plt.xticks(np.arange(min(epoch_numbers), max(epoch_numbers)+1, 1))
    plt.legend()
    plt.show()

    print('\n> Svoriai:')
    for k in range(0,n):
        print(W[k])

data = pd.read_csv('breast-cancer-wisconsin.data')
#prideti pavadinimas lengvesniam stulpeliu apdorojimui
data.columns = ['Sample_code', 'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
                'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin',
                'Normal_nucleoli', 'Mitoses','Class']

data = data.drop('Sample_code', axis=1)

#trukstamu duomenu uzpildymas mediana
data = data.replace('?',np.nan) 
#for col in data.columns:
#    print('\t%s: %d' % (col,data[col].isna().sum())) #yra tik 16 trukstamu reiksmiu ir jos visos 6 stulpelyje (Bare_nuclei)
data2 = data['Bare_nuclei']                  
data2 = pd.to_numeric(data2)
data2 = data2.fillna(data2.median())
data['Bare_nuclei'] = data2  

#standartizavimas skirtas isskirciu pasalinimui, ziurint pagal standartini nuokrypi nuo vidurkio!!!
Z = (data-data.mean())/data.std()
Z2 = Z.loc[((Z > -3).sum(axis=1)==Z.shape[1]) & ((Z <= 3).sum(axis=1)==Z.shape[1]),:]
data = data.loc[Z2.index]

#klasiu labels pakeitimas
data['Class'] = data['Class'].replace(2, 0)
data['Class'] = data['Class'].replace(4, 1)

#normavimas (be klases)
for col in data.columns:
    if col == 'Class':
        continue
    data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

train, validate, test = np.split(
    data.sample(frac=1, random_state=8), [int(0.7 * len(data)), int(0.85 * len(data))]  #daliname: pirmas padalijimas eina iki 0.8 viso dataset ilgio
                                                                                        #antras padalijimas eina nuo 0.8 iki 0.9 ilgio
                                                                                        #trecias nuo 0.9 iki galo
)
#neuronas mokomas naudojant paketini gradientini nusileidima
stochastinis_GD(train, validate, test)
