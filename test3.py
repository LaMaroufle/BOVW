import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

def distance (a, b): #Definition distance L1 entre 2 arrays de meme longueur
	dist = sum(abs(a-b))
	return dist

nbImg=20; # Nombre d'images par folder a prendre en compte
dbPath="./dataset/Learn" # chemin database contenant les dossiers categories
desPath="./Desc" # chemin de sauvegarde des descripteurs
dataPath="./Data"
histPath="./Hist"
nbClasse=5000

# On verifie si la BDD existe
if os.path.exists(dataPath + '/desfinal'):
	if os.path.exists(dataPath + '/labels') and os.path.exists(dataPath + '/centers'):
		print('Centers et Lables trouves !')
	else:
		print('Chargement de ' + dataPath + "/desfinal")
		desfinal=cPickle.load(open(dataPath + "/desfinal"))
		print("Descriptors loaded from " + dataPath + "/desfinal")
# Sinon on la cree
else:
	dir=os.listdir(dbPath)
	for i in range(0,len(dir)):
		dire= dbPath + '/' + dir[i]
		print("\nTraitement de "+ str(nbImg) +" images de : " + dire)

		# Preparation des variables de barre de chargement
		k=0
		nbDes=0
		# fin de pretaration
		for f in path(dire).walkfiles():
			if k<nbImg:
				# On verifie que les desc de l'image f ne sont pas deja calcules
				if os.path.exists(desPath + '/desc-' + dir[i] + str(k+1)):
					if k==nbImg-1:
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
							nbDes=nbDes + des.shape[0]

							# Incrementation et affichage chargement
							k=k+1
							sys.stdout.write('\r' + 'Chargement : ' + str(k) + '/' + str(nbImg) + ' Nombre de descripteurs : ' + str(nbDes) + ' soit ' + str(nbDes/k) + ' desc/img')
							sys.stdout.flush()
							# Fin chargement

							# Sauvegarde desc actuels
							cPickle.dump(des, open(desPath + "/desc-" + dir[i] + str(k), "wb"))

						except:
							# Si l'image pose probleme, on l'ignore et on passe a la suivante.
							print('Le fichier : ' + f + ' est introuvable ou invalide.')

	# Chargement des descripteurs enregistres
	k=1
	desfinal=np.array([])
	print('\nChargement des descripteurs...')
	nbImg=len(os.listdir(desPath))
	for f in path(desPath).walkfiles():
		try:
			Des=cPickle.load(open(f,'rb'))
			# Concatenation des descripteurs
			desfinal = np.append(desfinal, Des)
			desfinal = np.reshape(desfinal, (len(desfinal)/128, 128))
			nbDes=desfinal.shape[0]
		except:
			# Erreur : on recalcule le descripteur
			print('Le fichier ' + f + ' est invalide, il sera ignore.')

		#Chargement
		sys.stdout.write('\r' + 'Chargement : ' + str(k) + '/' + str(nbImg) + ' Nombre de descripteurs : ' + str(nbDes) + ' soit ' + str(nbDes/k) + 'desc/img')
		sys.stdout.flush()
		k=k+1

	print('\nEnregistrement de la liste des descripteurs...')
	cPickle.dump(desfinal, open(dataPath + "/desfinal", 'wb'))

if not os.path.exists(dataPath + '/labels') or not os.path.exists(dataPath + '/centers'):
	desfinal = np.float32(desfinal)

	# Preparation critere arret kmeans : iterations et epsilon
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	# Set flags (Just to avoid line break in the code)
	flags = cv2.KMEANS_RANDOM_CENTERS

	# Apply KMeans
	print("\nApplying kmeans...")
	ret,labels,centers = cv2.kmeans(desfinal,nbClasse,criteria,10,flags)
	desfinal=None

	print('Sauvegarde de labels...')
	cPickle.dump(labels, open(dataPath + "/labels", 'wb'))
	print('Sauvegarde de centers...')
	cPickle.dump(centers, open(dataPath + "/centers", 'wb'))
else:
	print('Chargement de centers...')
	centers=cPickle.load(open(dataPath + "/centers", 'rb'))
	print('Chargement de labels...')
	labels=cPickle.load(open(dataPath + "/labels", 'rb'))

dir=os.listdir(dbPath)
for i in range(0,len(dir)):
	dire= dbPath + '/' + dir[i]
	print("\nTraitement de "+ dire)

	k=0
	nb=0

	for f in path(dire).walkfiles():
		if k<nbImg:
			if os.path.exists(histPath + '/hist-' + dir[i] + str(k+1)):				#Test existence des histogrammes
				if k==nbImg-1:
					print('Histograms already computed.')
					print('Jumping to next folder...')
				k=k+1
			else:																		#Creation si necessaire de l'histogramme correspondant e l'image en cours
				descriptors=cPickle.load(open(desPath + '/desc-' + dir[i] + str(k+1))) #Chargement des descripteurs de cette image
				histo = np.zeros(nbClasse)
				charge=0
				for l in descriptors:													#Calcul pour chaque descripteur du centre le plus proche (toute la boucle for)
					min = distance(l,centers[0])										#et ajout de +1 dans histo a l'indice correspondant a ce centre
					indice = 0
					for j in range(nbClasse):
						buf=distance(l,centers[j])
						if (buf<min):
							min = buf
							indice = j
					histo[indice] += 1
					# Incrementation et affichage chargement
					charge += 1
					sys.stdout.write('\r' + 'Chargement : ' + str(charge*100/nbClasse) + '%' + ' de ' + str(k+1) + '/' + str(nbImg))
					sys.stdout.flush()
					# Fin chargement
				k=k+1
				cPickle.dump(histo, open(histPath + "/hist-" + dir[i] + str(k), "wb"))	#Sauvegarde de l'histogramme en cours

# Debut du SVM
# svc = svm.SVC(kernel='linear')
# svc.fit(labels, centers) # learn form the data
# SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False, shrinking=True, tol=0.001)
