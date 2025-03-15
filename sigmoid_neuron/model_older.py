import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# this works if the dataset has been downloaded locally
print("Extracting the dataset...")
data = pd.read_csv('breast-cancer-wisconsin.data')
# adding labels to the columns (missing in the imported CSV)
data.columns = ['Sample_code', 'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
                'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin',
                'Normal_nucleoli', 'Mitoses','Class']

data = data.drop('Sample_code', axis=1)
#print('Number of instances = %d' % (data.shape[0]))
#print('Number of attributes = %d' % (data.shape[1]))

data = data.replace('?',np.NaN)

#print('Number of missing values:')
#for col in data.columns:
#    print('\t%s: %d' % (col,data[col].isna().sum())) #yra tik 16 trukstamu reiksmiu ir jos visos 6 stulpelyje

data2 = data['Bare_nuclei']                         #naudota del pandas duomenu slicing 
data2 = pd.to_numeric(data2)
data2 = data2.fillna(data2.median())

data['Bare_nuclei'] = data2  

#data.boxplot(figsize=(20,3))
#plt.show()

#standartizavimas skirtas isskirciu pasalinimui, ziurint pagal standartini nuokrypi nuo vidurkio!!!
Z = (data-data.mean())/data.std()
Z2 = Z.loc[((Z > -3).sum(axis=1)==Z.shape[1]) & ((Z <= 3).sum(axis=1)==Z.shape[1]),:]
data = data.loc[Z2.index]

#normalizavimas skirtas tam, kad modeli butu lengviau optimizuoti, jis butu maziau jautrus pokyciams
#svoriu svyravimams

#y = data.pop('Class')
data['Class'] = data['Class'].replace(2, 0)
data['Class'] = data['Class'].replace(4, 1)

#normavimas (be klases, kadangi tai tiesiog 0 ir 1 reiksmes)
for col in data.columns:
    if col == 'Class':
        continue
    data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

#data.boxplot()
#plt.show()


#dataset dalijimas i 
#sample taip pat ismaiso dataset
train, test, validate = np.split(
    data.sample(frac=1, random_state=42), [int(0.8 * len(data)), int(0.9 * len(data))]  #daliname: pirmas padalijimas eina iki 0.8 viso dataset ilgio
                                                                                        #antras padalijimas eina nuo 0.8 iki 0.9 ilgio
                                                                                        #trecias nuo 0.9 iki galo
)

# Display different sets
#print("Training set:\n", train, "\n")
#print("Testing set:\n", test, "\n")
#print("Validation set:\n", validate)

#neuronas mokomas naudojant paketini gradientini nusileidima

totalError = 999999
epoch = 0
minError = 0.8
epochs = 100
learning_rate = 0.1

while (totalError > minError and epoch < epochs):
    