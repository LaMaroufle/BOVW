import cv2
import numpy as np
import os
import sys
from path import path
import cPickle
import matplotlib.pyplot as plt

dbPath="./dataset/paris" # chemin database contenant les dossiers categories
desPath="./Desc" # chemin de sauvegarde des descripteurs
dataPath="./Data"
imgPath='/louvre/paris_louvre_000032.jpg'

def L1(liste1, liste2):
	d = 0
	for i in range(0,127):
		d = d + abs(liste1[i] - liste2[i])
	return d

if not os.path.exists(dataPath + '/dist'):
	if not os.path.exists(dataPath + '/labels') or not os.path.exists(dataPath + '/centers'):
		print('labels ou centers manquant, lancer test3.py pour les creer')
	else:
		print('Chargement de centers...')
		centers=cPickle.load(open(dataPath + "/centers", 'rb'))

	img = cv2.imread(dbPath + imgPath,1)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT()
	kp,des = sift.detectAndCompute(gray,None)

	dist=np.array([])
	i=0
	print ("\nCalcul des distances :")
	for row in des:
		sys.stdout.write('\r' + 'Traitement du descripteur : ' + str(i+1) + '/' + str(des.shape[0]))
		sys.stdout.flush()
		distImgCent=[]
		for cent in centers:
			distImgCent.append(sum(abs(row-cent)))
		dist = np.append(dist, distImgCent.index(min(distImgCent)))
		i=i+1

	print('\nEnregistrement des distances...')
	cPickle.dump(dist, open(dataPath + '/dist', "wb"))

else:
	print('Distances trouve, chargement...')
	dist=cPickle.load(open(dataPath + "/dist", 'rb'))

print(dist)
plt.hist(dist,5000)
plt.title("Histograme de " + dbPath + imgPath)
plt.xlabel("Classes")
plt.ylabel("Nombre")
plt.show()

#distImgCent.index(min(distImgCent))
