# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

# Rendu de Khaled ABDRABO (p1713323) et Jean BRIGNONE (p1709655)

import gzip, numpy, torch
    
if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.05 # taux d'apprentissage
	nb_neurones_cc = 10 # nombre de neurones dans la couche cachée 
	
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	#2cc/200neurones/0.01 eta
	#3cc/10neurones/0.05 eta


	# on initialise le modèle et ses poids
	model = torch.nn.Sequential(
		torch.nn.Linear(data_train.shape[1], nb_neurones_cc),
		torch.nn.Sigmoid(),
		torch.nn.Linear(nb_neurones_cc, nb_neurones_cc),
		torch.nn.Sigmoid(),
		torch.nn.Linear(nb_neurones_cc, nb_neurones_cc),
		torch.nn.Sigmoid(),
		torch.nn.Linear(nb_neurones_cc, label_train.shape[1])
	)

	# init les poids du modèle
	torch.nn.init.uniform_(model[0].weight, -0.001, 0.001)
	torch.nn.init.uniform_(model[2].weight, -0.001, 0.001)
	torch.nn.init.uniform_(model[4].weight, -0.001, 0.001)
	torch.nn.init.uniform_(model[6].weight, -0.001, 0.001)

	# on initiliase l'optimiseur
	loss_func = torch.nn.MSELoss(reduction='sum')
	optim = torch.optim.SGD(model.parameters(), lr=eta)

	for n in range(nb_epochs):
		# on lit toutes les données d'apprentissage
		for x,t in train_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on met à jour les poids
			loss = loss_func(t, y)
			loss.backward()
			optim.step()
			optim.zero_grad()
			
		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for x,t in test_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on regarde si la sortie est correcte
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# on affiche le pourcentage de bonnes réponses
		print(acc/data_test.shape[0] * 100)
