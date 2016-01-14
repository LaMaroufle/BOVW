import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

def somme (a): #Definition distance L1 entre 2 arrays de meme longueur
	i=0
	sme=0
	for i in range(len(a)):
		sme+=a[i]
	return sme

nbImg=20; # Nombre d'images par folder a prendre en compte
dbPath="./dataset/paris" # chemin database contenant les dossiers categories
desPath="./Desc" # chemin de sauvegarde des descripteurs
dataPath="./Data"
histPath="./Hist"
nbClasse=5000

k=1
hists=np.array([])
print('\nChargement des histogrammes...')
for f in path(histPath).walkfiles():
	try:
		hist=cPickle.load(open(f,'rb'))
		hist=hist/somme(hist)
		# Concatenation des histogrammes
		hists = np.append(hists, hist)
		hists = np.reshape(hists, (len(hists)/nbClasse, nbClasse))
		nbHist=hists.shape[0]
	except:
		# Erreur : on ignore l'hist
		print('Le fichier ' + f + ' est invalide, il sera ignore.')

	#Chargement
	sys.stdout.write('\r' + 'Chargement : ' + str(k) + '/' + str(nbHist))
	sys.stdout.flush()
	k=k+1

print('\nEnregistrement de la liste des histogrammes...')
cPickle.dump(hists, open(dataPath + '/Hists', 'wb'))
