import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

dataPath="./Data"
desPath="./Desc"
nbClasse=5000
desChoice='/desc-moulinrouge6'

print('Chargement de la BDD d\'Histograme...')
hists=cPickle.load(open(dataPath + '/Hists','rb'))

def distance (a, b):
	dist = sum(abs(a-b))
	return dist

def somme (a): #Definition distance L1 entre 2 arrays de meme longueur
	i=0
	sme=0
	for i in range(len(a)):
		sme+=a[i]
	return sme

histo = np.zeros(nbClasse)
charge=0

if os.path.exists('histo'):
	print('Chargement de l\'histogramme a tester...')
	histo=cPickle.load(open("histo", 'rb'))
else:
	print('Chargement du descripteur a tester...')
	des=cPickle.load(open(desPath + desChoice,'rb'))
	print('Chargement de centers...')
	centers=cPickle.load(open(dataPath + "/centers", 'rb'))
	for l in des:													#Calcul pour chaque descripteur du centre le plus proche (toute la boucle for)
		min = distance(l,centers[0])										#et ajout de +1 dans histo a l'indice correspondant a ce centre
		indice = 0
		for j in range(nbClasse):
			if (distance(l,centers[j])<min):
				min = distance(l,centers[j])
				indice = j
		histo[indice] += 1/(des.shape[0])

		# Incrementation et affichage chargement
		charge += 1
		sys.stdout.write('\r' + 'Chargement : ' + str(charge*100/nbClasse) + '%')
		sys.stdout.flush()
		# Fin chargement
	cPickle.dump(histo, open('histo', "wb"))	#Sauvegarde de l'histogramme en cours

min=distance(histo,hists[0])
min2=0
k=0
dist=[]
histTest=histo/somme(histo)
print('histogramme charge :')
print(histTest)
for h in hists:
	buf=distance(h,histTest)
	dist.append(buf)
	k+=1

print('matrice des similarites : faible = mieux')
print(dist)
