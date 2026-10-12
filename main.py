from os import read
import pandas as pd
import random
dataFrame1MonJeuDeDonneeDentrainement = pd.read_csv("donnevrai.csv")
dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].map({"safe":0,"dangerous": 1})
X=dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist()
Y=dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()
print(X[:5],Y[:5])

w= random.randint(-10,10) #poids
b= random.randint(-10,10) #biais

def Zpredict(x,w, b):

    z = w * x + b
    if z >= 0:
        return 1
    else:
        return 0
#for x, y in zip(X, Y):
    y_pred = Zpredict(x, w, b)
    erreur = y - y_pred
    print("x =",x," y réel =", y," prédiction y =",y_pred," erreur =",erreur)



def tempo(w,b,eta,epochs,x):
    print("vitesse",x)
    for epoch in range(epochs):
        erreurs = 0
        for x, y in zip(X, Y):
            y_pred = Zpredict(x, w, b)
            erreur = y - y_pred
            if erreur != 0:
                w += eta * erreur * x
                b += eta * erreur
                erreurs += 1
        #print(f"Époque {epoch+1} : w = {w:.2f}, b = {b:.2f}, erreurs = {erreurs}")
        if erreurs == 0:
            print("Apprentissage terminé.")
            y_pred = Zpredict(x, w, b)
            return eta, epoch + 1 , y_pred
    return eta, epoch + 1 , erreurs

for i in range (7) :
    eta = 0.6 # taux de correction
    epochs = 200 # max
    x= 67
    print("w :", w," b : ",b, "eta et epoch",tempo(w,b,eta,epochs,x))

#w : -3  b :  -6 eta et epoch (0.7, 11)
#w : -9  b :  -6 eta et epoch (0.3, 13)
#w : -4  b :  -5 eta et epoch (0.6, 5)