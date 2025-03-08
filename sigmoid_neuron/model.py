import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# this works if the dataset has been downloaded locally
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
# adding labels to the columns (missing in the imported CSV)
data.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']

data = data.drop('Sample code', axis=1)
#print('Number of instances = %d' % (data.shape[0]))
#print('Number of attributes = %d' % (data.shape[1]))

data = data.replace('?',np.NaN)

#print('Number of missing values:')
#for col in data.columns:
#    print('\t%s: %d' % (col,data[col].isna().sum())) #yra tik 16 trukstamu reiksmiu ir jos visos 6 stulpelyje

data2 = data['Bare Nuclei']                         #naudota del pandas duomenu slicing 
data2 = pd.to_numeric(data2)
data2 = data2.fillna(data2.median())

data['Bare Nuclei'] = data2  

#data.boxplot(figsize=(20,3))
#plt.show()

#standartizavimas skirtas isskirciu pasalinimui, ziurint pagal standartini nuokrypi nuo vidurkio!!!

Z = (data-data.mean())/data.std()
Z2 = Z.loc[((Z > -3).sum(axis=1)==Z.shape[1]) & ((Z <= 3).sum(axis=1)==Z.shape[1]),:]
data = data.loc[Z2.index]

#print('Number of rows after discarding outliers = %d' % data.shape[0])

#normalizavimas skirtas tam, kad modeli butu lengviau optimizuoti, jis butu maziau jautrus
#svoriu svyravimams

y = data.pop('Class')
y = y.replace(2, 0)
y = y.replace(4, 1)

print('Number of attributes = %d' % (data.shape[1]))

for col in data.columns:
    data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())

#data.boxplot()
#plt.show()