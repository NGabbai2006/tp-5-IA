from os import read
import pandas as pd
import random
dataFrame1MonJeuDeDonneeDentrainement = pd.read_csv("donnevrai.csv")
dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].map({"safe":0,"dangerous": 1})
X=dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist()
Y=dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()
print(X[:5],Y[:5])
def Zpredict(x):
    z = w * x + b
    if z >= 0:
        return 1
    else:
        return 0
#for x, y in zip(X, Y):
    y_pred = Zpredict(x)
    erreur = y - y_pred
    print("x =",x," y réel =", y," prédiction y =",y_pred," erreur =",erreur)



def tempo(w,b,eta,epochs):
    for epoch in range(epochs):
        erreurs = 0
        for x, y in zip(X, Y):
            y_pred = Zpredict(x)
            erreur = y - y_pred
            if erreur != 0:
                w += eta * erreur * x
                b += eta * erreur
                erreurs += 1
        #print(f"Époque {epoch+1} : w = {w:.2f}, b = {b:.2f}, erreurs = {erreurs}")
    if erreurs == 0:
        print("Apprentissage terminé.")
    return eta, epoch + 1

for i in range (7) :
    w= random.randint(-10,10) #poids
    b= random.randint(-10,10) #biais
    eta = [0.1,0.5,0.5,0.7,0.8,0.2,1] # taux de correction
    epochs = 1000 # max
    print("w :", w," b :", "eta et epoch",tempo(w,b,eta[i],epochs))