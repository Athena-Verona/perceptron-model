import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sigmoid(a):
    return round(1/(1 + np.exp(-a)), 1)

def paketinis_GD(dataset, validation, test):
    W = []
    for i in range(0,9):
        random.seed(13)
        weight = random.uniform(-1,1)
        W.append(weight)

    y = dataset.pop('Class')
    y_val = validation.pop('Class')
    y_test = test.pop('Class')

    totalError = 999999
    totalErrorList = []
    totalErrorListVal = []
    epoch = 0
    y_pred = []

    minError = 0.2
    epochs = 200
    learning_rate = 0.99

    m = dataset.shape[0]
    m_val = validation.shape[0]
    m_test = test.shape[0]
    n = len(W)

    print("searching...")
    while (totalError > minError and epoch < epochs):

        gradSum = [0] * n
        totalError = 0

        for i in range(0, m): 

            a = (dataset.iloc[i] * W).sum() #daugina su visais eilutes W elementais
            yi = sigmoid(a)
            t = y.iloc[i]
            y_pred.append(round(yi, 1))

            for k in range(0, n): 
                gradSum[k] = gradSum[k] + (yi - t)*yi*(1 - yi)*dataset.iloc[i, k]

            error = (t - yi)**2
            totalError = totalError + error
        #accuracy/tikslumas

        for k in range(0, n): 
            W[k] = W[k] - learning_rate * (gradSum[k] / m)

        epoch+=1
        totalError=totalError/m
        totalErrorList.append(totalError)


        #o cia validavimas
        totalErrorVal=0
        for i in range(0, m_val): 

            a = (validate.iloc[i] * W).sum()
            yi = sigmoid(a)
            t = y_val.iloc[i]
            error = (t - yi)**2
            totalErrorVal = totalErrorVal + error

        totalErrorListVal.append(totalErrorVal/m_val)

    #testavimas
    totalError = 0
    for i in range(0, m_test): 

        a = (test.iloc[i] * W).sum() #daugina su visais eilutes W elementais
        yi = sigmoid(a)
        t = y_test.iloc[i]
        error = (t - yi)**2
        totalError = totalError + error
        
    totalError = totalError/m_test
    print('Testavimo paklaida :', totalError)

    epoch_numbers = list(range(1, epoch + 1))
    plt.plot(epoch_numbers, totalErrorList, color='blue')
    plt.plot(epoch_numbers, totalErrorListVal, color='red')

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
    data.sample(frac=1, random_state=42), [int(0.8 * len(data)), int(0.9 * len(data))]  #daliname: pirmas padalijimas eina iki 0.8 viso dataset ilgio
                                                                                        #antras padalijimas eina nuo 0.8 iki 0.9 ilgio
                                                                                        #trecias nuo 0.9 iki galo
)
#neuronas mokomas naudojant paketini gradientini nusileidima
paketinis_GD(train, validate, test)

plt.title('Paklaida')
plt.show()