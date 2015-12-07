import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

# On verifie si la BDD existe
if os.path.exists('filee'):
	dd=cPickle.load(open("file"))
	print("descriptors loaded from 'file'")
# Sinon on la cree
else:
	dir=os.listdir("./dataset/paris")
	for i in range(0,len(dir)):
		desfinal=np.array([])
		dire='./dataset/paris/' + dir[i]
		print("\ntraitement des image de : " + dire)

		# Preparation des variables de chargement
		k=0
		tai=len(os.listdir(dire))
		# fin de pretaration

		# On verifie l'existence des desc pour cette classe
		if os.path.exists('desc' + str(i)):
			print("descriptors already computed in 'file" + str(i) + "'")
			print("computing next folder...")

		# S'ils n'existent pas deja, on les calcule
		else:
			for f in path(dire).walkfiles():
				if f.endswith('jpg') and f!='NULL':
					img = cv2.imread(f,1)
					try:
						gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
						sift = cv2.SIFT()
						kp,des = sift.detectAndCompute(gray,None)
						desfinal = np.append(desfinal, des)
						desfinal = np.reshape(desfinal, (len(desfinal)/128, 128))

						# Incrementation et affichage chargement
						k=k+1
						sys.stdout.write('\r' + 'chargement : ' + str(k*100/tai) + '%')
						sys.stdout.flush()
						# Fin chargement

					except:
						# Si l'image pose probleme, on l'ignore et on apsse a la suivante, augmente le chargement
						k=k+1
						print('le fichier : ' + f + 'est introuvable ou invalide')

			# On enregistre les desc de la classe actuelle
			cPickle.dump(desfinal, open("desc" + str(i), "wb"))

	# Chargement des descripteurs enregistres
	for i in range(0,len(dir)):
		desfinal = np.append(desfinal, cPickle.load(open("desc" + str(i))))

	# Concatenation des descripteurs
	descriptors = np.array([])
	descriptors = np.append(descriptors, desfinal)

	# Preparation de la matrice de descripteurs pour le Kmeans
	desc = np.reshape(descriptors, (len(descriptors)/128, 128))
	desc = np.float32(desc)

	# Preparation critere arret kmeans : iterations et epsilon
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	# Set flags (Just to avoid line break in the code)
	flags = cv2.KMEANS_RANDOM_CENTERS

	# Apply KMeans
	print("applying kmeans...")
	ret,labels,centers = cv2.kmeans(desc,6000,criteria,10,flags)

	# concatenation desc et labels
	labels=np.matrix.transpose(labels)
	desfinal=np.matrix.transpose(desfinal)
	dd=np.vstack((labels,desfinal))
	dd=np.matrix.transpose(dd)

	# Sauvegarde descripteurs et labels
	cPickle.dump(dd, open("file", "wb"))
	print ("descriptors + labels saved in 'file'")

# Debut du SVM
# svc = svm.SVC(kernel='linear')
# svc.fit(labels, centers) # learn form the data
# SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False, shrinking=True, tol=0.001)
