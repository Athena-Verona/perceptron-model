import random
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def stepFunction(a):
    if a >= 0:
        return 1
    if a < 0:
        return 0

def sigmoid(a):
    return round(1/(1 + np.exp(-a)), 1)

#skiriamojo pavirsiaus funkcija pagal x2 (x2 = koordinate y)
def f(x,w1,w2,b):
    return -1*((w1*x+b)/w2)           

#cia generuojami 10 tiesiskai atskiriami taskai, parametrai: 
#samples = tasku kiekis, features = pozymiai, centers = tasku grupiu kiekis, random_state = atsitiktinumo zyme
#y = klase
#X = matrica (10,2) dydzio
X, y = make_blobs(n_samples=20, n_features=2, centers=2, random_state=38)

choice = input("Slenkstinei funkcijai spauskite 1, o sigmoidinei 2\n")
print("Searching...")

weights_array = []
#random.seed(4) #skirta gauti atkartotiniems rezultatams

#ciklas kartojamas trims svoriu rinkiniams rasti
for i in range(0,3):
    notFound = True

    while notFound: 

        #atsitiktiniai pasirinkimai su random 
        w1 = random.randint(-5, 5) #intervalas [-5;5] be 0
        w2 = random.randint(-5, 5)
        b = random.randint(-5, 5)

        #ciklas skirtas uztikrinti, kad visiems 20 tasku tinka svoriai
        for i in range(20):

            #i-tajam taskui skaiciuojamas a
            a = X[i,0]*w1 + X[i,1]*w2 + b

            #pagal vartotojo pasirinkima renkama aktyvacijos funkcija
            if int(choice) == 1:
                answer = stepFunction(a)
            
            if int(choice) == 2:
                answer = sigmoid(a)

            #jei f(a) nesutampa su is anksto aprasyta klase, ciklas nutraukiamas ir kartojamas su naujais svoriais
            if answer != y[i]:
                break

            #jei ciklas pasiekia 20-taji taska, skaitoma, kad svoriai sekmingi
            if i == 19:
                notFound = False
                print("Values found")
                weights_row = (w1,w2,b)
                print(w1,w2,b)

    weights_array.append(weights_row)

#vizualizacijos kodas

plt.scatter(X[:, 0], X[:, 1], c=y)      #x[eilute, stulpelis]

x = np.linspace(-10, 10, 1000)

colors = ['c','m','y']
Origins = []    #vektoriu pradiniu tasku koordiciu masyvas

#ciklas, kuris vizualizuoja skiriamuosius pavirsius ir apskaiciuoja vektoriu pradzios taskus pagal poslinki f(b)
for i in range(0,3):

    y1 = f(x, weights_array[i][0], weights_array[i][1], weights_array[i][2])       #y1 = f(x,w1,w2,b)
    plt.plot(x, y1, color=colors[i], label=i)

    Origin1 = f(weights_array[i][2], weights_array[i][0], weights_array[i][1], weights_array[i][2])
    Origins.append(Origin1) #solved for 0

#vektoriai sudaromi pagal svoriu masyva
Vectors = np.array([[weights_array[i][0], weights_array[i][1]] for i in range(3)])

#pradzios taskai yra (b, f(b))
origin = np.array([[weights_array[0][2],weights_array[1][2],weights_array[2][2]],[Origins[0],Origins[1],Origins[2]]]) #pradzios taskas

plt.quiver(*origin, Vectors[:,0], Vectors[:,1], color=['c','m','y'], scale=20)
plt.title("Du klasių klasteriai ir skiriamųjų paviršių tiesės")
plt.axis('equal')
plt.legend(['Klasė 0', 'Tiesė 1', 'Tiesė 2', 'Tiesė 3'])
plt.show()
