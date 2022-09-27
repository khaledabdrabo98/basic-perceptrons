# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

# Rendu de Khaled ABDRABO (p1713323) et Jean BRIGNONE (p1709655)

import gzip, numpy, torch

if __name__ == '__main__':
    # Hyperparamètres
    #
    # eta : responsable du taux de variation dans les poids.
    # Plus sa valeur est grande, plus l'algorithme converge rapidement avec des résultats moins précis.
    # Au contraire, plus sa valeur est faible, l'algorithme converge plus
    # lentement mais se rapproche des bonnes résultats.
    # Si la valeur est beaucoup trop grande, le modèle apprend trop vite et donc il finit par rien apprendre.
    #
    # w_min et w_max : sont les poids de départs. Quand ils se rapprochent de 0, l'algorithme converge plus rapidement.

    batch_size = 1  # nombre de données lues à chaque fois
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.001  # taux d'apprentissage
    w_min = -0.001  # poids minimum
    w_max = 0.001  # poids maximum

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on initialise le modèle et ses poids
    # taille du tenseur w = 784 (data_train.shape[1]) * 10 (label_train.shape[1]) = 28*28*10 = 7840
    w = torch.empty((data_train.shape[1], label_train.shape[1]), dtype=torch.float)
    # taille du tenseur b = 1*10 = 10
    b = torch.empty((1, label_train.shape[1]), dtype=torch.float)
    torch.nn.init.uniform_(w, w_min, w_max)
    torch.nn.init.uniform_(b, w_min, w_max)

    nb_data_train = data_train.shape[0]
    nb_data_test = data_test.shape[0]
    # taille du tenseur indices = 12600 (les 63000 / 5 [batch_size])
    indices = numpy.arange(nb_data_train, step=batch_size)

    for n in range(nb_epochs):
        # on mélange les (indices des) données
        numpy.random.shuffle(indices)
        # on lit toutes les données d'apprentissage
        for i in indices:
            # on récupère les entrées
            # taille du tenseur x = 5 * 784 = 3920
            x = data_train[i:i + batch_size]
            # on calcule la sortie du modèle
            # taille du tenseur y = 5 * 10 = 50
            y = torch.mm(x, w) + b
            # on regarde les vrais labels
            # taille du tenseur t = 5 * 10 = 50
            t = label_train[i:i + batch_size]
            # on met à jour les poids
            # taille du tenseur du gradiant = 5 * 10 = 50
            grad = (t - y)
            w += eta * torch.mm(x.T, grad)
            b += eta * grad.sum(axis=0)

        # test du modèle (on évalue la progression pendant l'apprentissage)
        # conteur des bonnes réponses
        acc = 0.
        # on lit toutes les données de test
        for i in range(nb_data_test):
            # on récupère l'entrée
            # taille du tenseur x = 1 * 784 = 784
            x = data_test[i:i + 1]
            # on calcule la sortie du modèle
            # taille du tenseur y = 1 * 10 = 10
            y = torch.mm(x, w) + b
            # on regarde le vrai label
            # taille du tenseur t = 1 * 10 = 10
            t = label_test[i:i + 1]
            # on regarde si la sortie est correcte
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        # on affiche le pourcentage de bonnes réponses
        print(acc / nb_data_test * 100)
