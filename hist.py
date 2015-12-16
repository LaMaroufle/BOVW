import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

dbPath="./dataset/paris" # chemin database contenant les dossiers categories
desPath="./Desc" # chemin de sauvegarde des descripteurs
dataPath="./Data"

def L1(liste1, liste2):
	d = 0
	for i in range(0,127):
		d = d + abs(liste1[i] - liste2[i])
	return d

if not os.path.exists(dataPath + '/labels') or not os.path.exists(dataPath + '/centers'):
	print('labels ou centers manquant, lancer test3.py pour les creer')
else:
	print('Chargement de centers...')
	centers=cPickle.load(open(dataPath + "/centers", 'rb'))

img = cv2.imread(dbPath + '/notredame/paris_notredame_000998.jpg',1)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp,des = sift.detectAndCompute(gray,None)

hist=np.array([])
distImgCent=np.array([])
i=0
j=0
print ("\nCalcul des distances :")
for row in des:
	sys.stdout.write('\r' + 'Traitement du descripteur : ' + str(i+1) + '/' + str(des.shape[0]))
	sys.stdout.flush()
	for cent in centers:
		distImgCent=L1(row,cent)
		j+1
	i+1

print(distImgCent)
#distImgCent.index(min(distImgCent))
