
from os import read
import pandas as pd
import random
dataFrame1MonJeuDeDonneeDentrainement = pd.read_csv("donnevrai.csv")
dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].map({"safe":0,"dangerous": 1})
X=dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist()
Y=dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()
print(X[:5],Y[:5])

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



def tempo(w,b,eta,epochs,x_test):

    for epoch in range(epochs):
        erreurs = 0
        for vitesse, y in zip(X, Y):
            y_pred = Zpredict(vitesse, w, b)
            erreur = y - y_pred
            if erreur != 0:
                w += eta * erreur * vitesse
                b += eta * erreur
                erreurs += 1
        #print(f"Époque {epoch+1} : w = {w:.2f}, b = {b:.2f}, erreurs = {erreurs}")
        if erreurs == 0:
            print("Apprentissage terminé.")
            y_pred = Zpredict(x_test, w, b)
            return y_pred
    y_pred = Zpredict(x_test, w, b)
    return y_pred



#w : -3  b :  -6 eta et epoch (0.7, 11)
#w : -9  b : -6 eta et epoch (0.3, 13)
#w : -4  b : -5 eta et epoch (0.6, 5)

from tkinter import Tk, Label , StringVar , Button, Entry
root = Tk() # Création de la fenêtre racine
txt = StringVar(root) # Création d'une variable de type StringVar qui recup la valeur
txt.set("Entrer une vitesse : ")
txt1 = StringVar(root) # Création d'une variable de type StringVar qui recup la valeur
label = Label(root, textvariable=txt) 
vitesse_entry = Entry(root)
def exe() :
    vitesse = int(vitesse_entry.get())
    w= -4 #random.randint(-10,10) #poids
    b= -5 #random.randint(-10,10) #biais    
    eta = 0.6 # taux de correction
    epochs = 200 # max
    y_pred = tempo(w,b,eta,epochs,vitesse)
    if y_pred == 1:
        txt1.set("La vitesse est dangereuse")
    else:
        txt1.set("La vitesse est sûre")
button = Button(root, text="Prédire", command=exe)
label.grid(column=0, row=0)
vitesse_entry.grid(column=1, row=0)
button.grid(column=2, row=0)
widget = Label(root, text="Résultat : ",textvariable=txt1) # Création d'un widget Label pour afficher le résultat
widget.grid(column=0, row=1)
root.mainloop() # Lancement de la boucle principal