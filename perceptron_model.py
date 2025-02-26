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


def f(x,w1,w2,b):
    return -1*((w1*x+b)/w2)           

#generarate 10 linearly seperable (tiesiskai atskiriami) points
#samples = tasku kiekis, features = pozymiai, centers = tasku grupiu kiekis
#y = which class point belongs to
#X = samples
X, y = make_blobs(n_samples=20, n_features=2, centers = 2, random_state=8)

# taking two inputs at a time
choice = input("Slenkstinei funkcijai spauskite 1, o sigmoidinei 2\n")

print("Searching...")

weights_array = []
random.seed(4) 

for i in range(0,3):
    notFound = True

    while notFound: 
        w1 = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4,]) #nezinia, ar gali buti svoris = 0
        w2 = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4,]) #skiriamajam pavirsiui negalima dalyba is 0 (ziureti f(x))
        b = random.randint(-5, 5)
        for i in range(20):
            a = X[i,0]*w1 + X[i,1]*w2 + b

            if int(choice) == 1:
                answer = stepFunction(a)
            
            if int(choice) == 2:
                answer = sigmoid(a)

            if answer != y[i]:
                break
            if i == 19:
                notFound = False
                print("Values found")
                weights_row = (w1,w2,b)
                print(w1,w2,b)

    weights_array.append(weights_row)

plt.scatter(X[:, 0], X[:, 1], c=y)      #x[row,column]


x = np.linspace(-10, 10, 1000)

colors = ['c','m','y']
Origins = []

for i in range(0,3):
    y1 = f(x, weights_array[i][0], weights_array[i][1], weights_array[i][2])       #y1 = f(x,w1,w2,b)
    plt.plot(x, y1, color=colors[i])

    Origin1 = f(weights_array[i][2], weights_array[i][0], weights_array[i][1], weights_array[i][2])
    Origins.append(Origin1) #solved for 0

Vectors = np.array([[weights_array[i][0], weights_array[i][1]] for i in range(3)])
origin = np.array([[weights_array[0][2],weights_array[1][2],weights_array[2][2]],[Origins[0],Origins[1],Origins[2]]]) #pradzios taskas

plt.quiver(*origin, Vectors[:,0], Vectors[:,1], color=['c','m','y'], scale=20)

plt.title("Dvi tasku klases ir skiriamieji pavirsiai")
plt.axis('equal')
plt.show()
