import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

class err(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

NbImg=30; # Nombre d'images par folder a prendre en compte
DBpath="./dataset/paris" # chemin database contenant les dossiers categories
SavePath="./Desc" # chemin de sauvegarde des descripteurs

# On verifie si la BDD existe
if os.path.exists(SavePath + '/desfinal'):
	dd=cPickle.load(open(SavePath + "desfinal"))
	print("Descriptors loaded from " + SavePath + "/desfinal")
# Sinon on la cree
else:
	dir=os.listdir(DBpath)
	for i in range(0,len(dir)):
		desfinal= np.array([])
		dire= DBpath + '/' + dir[i]
		print("\nTraitement de "+ str(NbImg) +" images de : " + dire)

		# Preparation des variables de barre de chargement
		k=0
		NbDesc=0
		# fin de pretaration
		for f in path(dire).walkfiles():
			if k<30:
				# On verifie que les desc de l'image f ne sont pas deja calcules
				if os.path.exists(SavePath + '/desc-' + dir[i] + str(k+1)):
					if k==29:
						print('Descriptors already computed.')
						print('Jumping to next folder...')
					k=k+1
				# Si les desc de celle image ne sont pas calcules :
				else:
					if f.endswith('jpg') and f!='NULL':
						try:
							img = cv2.imread(f,1)

							# Calcul des desc de chaque image
							gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
							sift = cv2.SIFT()
							kp,des = sift.detectAndCompute(gray,None)
							NbDesc=NbDesc + des.shape[0]

							# Incrementation et affichage chargement
							k=k+1
							sys.stdout.write('\r' + 'Chargementt : ' + str(k) + '/' + str(NbImg) + ' Nombre de descripteurs : ' + str(NbDesc) + ' soit ' + str(NbDesc/k) + ' desc par image')
							sys.stdout.flush()
							# Fin chargement

							# Sauvegarde desc actuels
							cPickle.dump(des, open(SavePath + "/desc-" + dir[i] + str(k), "wb"))

						except err as e:
							# Si l'image pose probleme, on l'ignore et on passe a la suivante.
							print('Le fichier : ' + f + ' est introuvable ou invalide.')

	# Chargement des descripteurs enregistres
	k=1
	print('\nChargement des descripteurs...')
	for f in path(SavePath).walkfiles():
		sys.stdout.write('\r' + 'Chargement : ' + str(k) + '/' + str(len(os.listdir(SavePath))))
		sys.stdout.flush()
		try:
			desfinal = np.append(desfinal, cPickle.load(open(f,'rb')))
		except err as e:
			# Erreur : on recalcule le descripteur
			print('Le fichier ' + f + ' est invalide, il sera ignore.')
		k=k+1

	cPickle.dump(desfinal, open(SavePath + "/desfinal", 'wb'))

# Preparation de la matrice de descripteurs pour le Kmeans
desc = np.reshape(desfinal, (len(desfinal)/128, 128))
desc = np.float32(desc)

# Preparation critere arret kmeans : iterations et epsilon
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
print("\n\nApplying kmeans...")
ret,labels,centers = cv2.kmeans(desc,6000,criteria,10,flags)

# Debut du SVM
# svc = svm.SVC(kernel='linear')
# svc.fit(labels, centers) # learn form the data
# SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False, shrinking=True, tol=0.001)
