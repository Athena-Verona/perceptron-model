import random
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def stepFunction(a, i):
    if a >= 0:
        return 1
    if a < 0:
        return 0

def search(X,result):
    #random.seed(2)  
    print("Searching...")
    notFound = True

    while notFound: 
        w1 = random.randint(-5, 5)
        w2 = random.randint(-5, 5)
        b = random.randint(-5, 5)

        for i in range(20):
            a = X[i,0]*w1 + X[i,1]*w2 + b
            answer = stepFunction(a, i)
            if answer != result[i]:
                break
            if i == 19:
                notFound = False
                print("Values found")
                print(w1,w2,b)

def f(x,w1,w2,b):
    return -1*((w1*x+b)/w2)           

#generarate 10 linearly seperable (tiesiskai atskiriami) points
#samples = tasku kiekis, features = pozymiai, centers = tasku grupiu kiekis
#y = which class point belongs to
#X = samples
X, y = make_blobs(n_samples=20, n_features=2, centers = 2, random_state=42) #38

#search(X,y)
#random.seed(2)  
print("Searching...")
notFound = True
while notFound: 
    w1 = random.randint(-5, 5)
    w2 = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) #skiriamajam pavirsiui negalima dalyba is 0 (ziureti f(x))
    b = random.randint(-5, 5)
    for i in range(20):
        a = X[i,0]*w1 + X[i,1]*w2 + b
        answer = stepFunction(a, i)
        if answer != y[i]:
            break
        if i == 19:
            notFound = False
            print("Values found")
            print(w1,w2,b)


x = np.linspace(-5, 10, 1000)


#plt.show()
#print(y)
plt.scatter(X[:, 0], X[:, 1], c=y)      #x[row,column]
plt.plot(x, y1)
plt.title("Dvi tasku klases")
plt.show()

#print('Enter your name:')
#x = input()
#print('Hello, ' + x) 