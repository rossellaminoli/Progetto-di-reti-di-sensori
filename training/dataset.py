import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
import os

#Questa classe è progettata per gestire il caricamento e la preparazione dei dati per un progetto di machine learning.
class CSVDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4, k_folds=3, window_size=1, overlap=1, sample_rate=60):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k_folds

        self.window_size = window_size #in secondi
        self.overlap = overlap #sovrapposizione tra le finestre, parametro che noi dovremo modificare
        self.sample_rate = sample_rate #freq di campionamento

        self.legend = None#Inizializza una variabile per le etichette delle classi
        self.x, self.y = None, None#Variabili per memorizzare i dati di input e le etichette


        #Liste per memorizzare gli indici di training e di validazione
        self.train_indices = []
        self.val_indices = []

    #funzione essenziale per preparare i dati per l'addestramento e la validazione incrociata K-fold
    def setup(self, stage=None):
        self.x, self.y = self.create_dataset()
        k_fold = KFold(n_splits=self.k, shuffle=True, random_state=42)

        #Questo ciclo permette di preparare i dati per l'addestramento e la validazione incrociata K-fold, memorizzando gli indici necessari per ciascun fold
        for train_idx, val_idx in k_fold.split(self.x):
            self.train_indices.append(train_idx)
            self.val_indices.append(val_idx)

    #funzione utilizzata per suddividere i dati in finestre temporali, ciascuna con una propria etichetta
    def windowing(self, data):
        windows, labels = [], []

        for i in range(0, len(data) - self.window_size * self.sample_rate, int(self.overlap * self.sample_rate)):
            window = data.iloc[i:i + self.window_size * self.sample_rate]
            label = window["label"].iloc[0]#Prende l'etichetta associata alla finestra
            window = window.drop(columns=["label"])#Rimuove la colonna delle etichette dai dati della finestra.
            windows.append(window.values)
            labels.append(label)

        #Converte le liste di finestre e etichette in array numpy per un'elaborazione più efficiente.
        return np.array(windows), np.array(labels)

    #caricamento e del preprocessing dei dati da file CSV per creare il dataset utilizzato durante l'addestramento e la validazione.
    def create_dataset(self):
        x, y = [], [] #liste utilizzate per memorizzare i dati di input e le etichette rispettivamente.
        for file in os.listdir(self.root_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(#Legge il file CSV e lo carica in un DataFrame df.
                    os.path.join(self.root_dir, file),
                    index_col=0,
                    header=None,
                    names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                )
                df["label"] = file.split("_")[1].split(".")[0]#Estrae l'etichetta dal nome del file
                x_i, y_i = self.windowing(df)#Chiama il metodo windowing per dividere il DataFrame in finestre di dati e etichette corrispondenti.
                x.extend(x_i)#aggiunge gli elementi di x_i alla lista x
                y.extend(y_i)

        x, y = np.array(x), np.array(y)#Converte le liste x e y in array numpy per una gestione più efficiente dei dati.

        # Label encoding -> codifica le etichette e memorizza la legenda delle etichette e le etichette codificate
        self.legend, y = self.labels_encoding(y)

        return x, y #x=dati in input, y=etichette

    #funzione che codifia le etichette
    #Il metodo è statico e non dipende da alcun attributo dell'istanza della classe
    #y -> array delle etichette che deve essere codificato
    @staticmethod
    def labels_encoding(y):
        categories, inverse = np.unique(y, return_inverse=True)#Restituisce le categorie uniche presenti in y e un array di indici inversi.

        # Create the one-hot encoded matrix
        one_hot = np.zeros((y.size, categories.size))#Crea una matrice di zeri con dimensioni pari al numero di etichette (y.size) e al numero di categorie uniche (categories.size).
        one_hot[np.arange(y.size), inverse] = 1#Imposta gli elementi appropriati della matrice a 1 per creare la codifica one-hot.

        return categories, one_hot.astype(np.float64) #restituisce le categorie uniche trova in y e restituisce la matrice in formato float64

    # Questo metodo è responsabile della preparazione di un dataset di PyTorch
    #index ->Un elenco di array di indici, uno per ciascun fold
    def prepare_dataset(self, indexes, fold):
        #tensore in PyTorch è una struttura dati fondamentale per rappresentare e manipolare i dati all'interno dei modelli di deep learning, sono simili agli array di numpy
        #converte dati in input e etichette in tensori
        x = torch.tensor(self.x[indexes[fold]], dtype=torch.float32)
        y = torch.tensor(self.y[indexes[fold]], dtype=torch.float32)

        return torch.utils.data.TensorDataset(x, y)#Crea un oggetto TensorDataset di PyTorch che contiene i tensori di input x e le etichette y.


    #funzione utilizzata per creare un dataloader per il set di addestramento, specifico per un determinato fold
    def train_dataloader(self, fold=0):
        return DataLoader(
            self.prepare_dataset(self.train_indices, fold, ),#Chiama il metodo prepare_dataset per ottenere un dataset di training specifico per il fold corrente
            batch_size=self.batch_size,#Imposta la dimensione del batch, che è stata definita nell'inizializzatore della classe.
            num_workers=self.num_workers,#Imposta il numero di worker per il caricamento dei dati, definito nell'inizializzatore della classe.
            shuffle=True,#Mescola i dati ad ogni epoca, per garantire che il modello non impari l'ordine dei dati.
            persistent_workers=True#Mantiene i worker attivi tra le epoche, migliorando le prestazioni quando si utilizzano molti worker.
        )

    #funzione per creare un dataloader per il set di validazione, specifico per un determinato fold nella validazione incrociata K-fold.
    def val_dataloader(self, fold=0):
        return DataLoader(
            self.prepare_dataset(self.val_indices, fold),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,#Non mescola i dati, poiché durante la validazione è importante mantenere l'ordine dei dati.
            persistent_workers=True
        )
