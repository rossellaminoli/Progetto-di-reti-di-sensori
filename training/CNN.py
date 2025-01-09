from pathlib import Path
import torch
import torch.nn.functional as f
import lightning as pl
from sklearn.metrics import precision_score, f1_score, confusion_matrix #metriche per valutare un modello
import torch.nn as nn
import numpy as np

from utils.utility import cm_analysis

#rete CNN, c'è una fase di pre-processing
class CNN(pl.LightningModule):

    def __init__(self, input_dim, fold, classes_names, output_dim=2, learning_rate=1e-3):
        super(CNN, self).__init__() #chiama il costruttore della super classe
        self.name = "CNN" #nome del modello
        self.classes_labels = classes_names #etichette delle classi
        self.fold = fold #identifica il fold corrente
        self.classes = output_dim #num di classi di output

        #liste per memorizzare le predizioni e i target di validazione e test
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        self.learning_rate = learning_rate #tasso di apprendimento
        self.loss_function = nn.CrossEntropyLoss() #funzione di perdita

        self.input_dim = 6 #num di canali di input
        self.dim = 64 #num di canali di output
        self.filter_size = 3 #dim del filtro convoluzionale
        self.window_size = input_dim #dim della finestra di input

        # Convolution Branch
        self.conv1 = nn.Sequential( #nn.Sequential è una classe di PyTorch che permette di costruire un modello come una sequenza di layer, disposti uno dopo l'altro
            nn.Conv1d(self.window_size, self.dim, self.filter_size),#Funzione: Applicare una convoluzione sull'input per estrarre caratteristiche rilevanti.
            nn.BatchNorm1d(self.dim),#Funzione: Stabilizzare e velocizzare l'addestramento riducendo le variazioni nei valori delle attivazioni.
            nn.PReLU(),#Funzione: Introduce non-linearità nel modello, permettendo di apprendere relazioni complesse nei dati.
            nn.MaxPool1d(2),#Funzione: Riduce la dimensionalità del data, mantenendo solo i valori massimi e riducendo il numero di parametri e il rischio di overfitting.
            nn.Dropout(p=0.1)#Funzione: Prevenire l'overfitting disattivando casualmente una percentuale dei neuroni durante l'addestramento, p è una probabilità
        )  # (_, 64, 2)

        #definisce la parte completamente connessa (fully connected) della rete neurale.
        self.fc1 = nn.Sequential(
            nn.Linear(self.dim * 2, 128),#layer che apllica trasf. lineare, self.dim*2=dati ingresso, 128=dati in uscita
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),#Trasformare il tensore di input in un tensore con output_dim elementi, preparandolo per la fase di classificazione finale
            nn.Softmax(dim=-1)#Funzione: Convertire i valori in probabilità per ciascuna classe di output.
        )

    def id(self):
        return f"{self.name}_{self.window_size}"

    def forward(self, x):
        #x input con dimensioni: (batch size, d_model, length)
        # x = x.view(-1, self.input_dim, self.window_size)
        x = self.conv1(x)

        #view viene usata per rimodellare x in una matrice bidimensionale
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    #funzione che definisce un singolo passo di addestramento nella rete neurale
    def training_step(self, batch, batch_idx):
        x, y = batch #Un singolo batch di dati, generalmente un tuple (x, y) dove x sono i dati di input e y sono le etichette corrispondenti.
        y_hat = self(x) #ottiene le predizioni
        #y = torch.tensor(y, dtype=torch.long) if isinstance(y, np.ndarray) else y #
        loss = self.compute_loss(y_hat, y) #calcola la perdita
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #registra la perdita di addestramento
        return loss

    # utilizzata per eseguire un singolo passo di validazione nel processo di addestramento di una rete neurale.
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #y = torch.tensor(y, dtype=torch.long) if isinstance(y, np.ndarray) else y #
        val_loss = self.compute_loss(y_hat, y)#perdita di validazione
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Collect predictions and targets -> raccolta predizioni e target
        self.val_predictions.append(np.argmax(y_hat.cpu().numpy(), 1))
        self.val_targets.append(np.argmax(y.cpu().numpy(), 1))

        return val_loss

    #viene chiamata alla fine di ogni epoca di validazione per valutare le prestazioni del modello e registrare metriche importanti.
    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets from this epoch
        val_predictions = np.concatenate(self.val_predictions)
        val_targets = np.concatenate(self.val_targets)

        # Log or print confusion matrix and classification report
        precision = precision_score(val_targets, val_predictions, average='macro', zero_division=0)
        f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=0)
        self.log("prec_macro", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("f1_score", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear stored values for the next epoch
        self.val_predictions.clear() #svuota la lista delle predizioni di validazione per preparasi alla nuova epoca
        self.val_targets.clear()#stessa cosa ma con i target

    #utilizzata per eseguire un singolo passo di test nel processo di valutazione del modello
    #Questa funzione è simile alla funzione di validazione, ma è specificamente progettata per valutare il modello sui dati di test.
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #y = torch.tensor(y, dtype=torch.long) if isinstance(y, np.ndarray) else y#
        test_loss = self.compute_loss(y_hat, y) #perdita di test
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Collect predictions and targets
        self.test_predictions.append(np.argmax(y_hat.cpu().numpy(), 1))
        self.test_targets.append(np.argmax(y.cpu().numpy(), 1))

        return test_loss

    def on_test_end(self):
        output_path = f"output/{self.id()}" #crea percorso di output
        Path(output_path).mkdir(parents=True, exist_ok=True) #crea la directory di output

        test_predictions = np.concatenate(self.test_predictions)
        test_target = np.concatenate(self.test_targets)
        #analisi della matrice di confusione
        cm_analysis(#crea una matrice di confusione
            test_target,
            test_predictions,
            f"{output_path}/confusion_matrix_segments_fold_{self.fold}", #percorso del file dove salvare la matrice di confusione
            range(self.classes),
            self.classes_labels,
            specific_title=f"Segments: {self.id()} fold {self.fold}" #titolo specifico per la matrice di confusione
        )
        self.fold += 1 #incrementa il fold per passare al fold successivo

    #definisce l'ottimizzatore utilizzato per aggiornare i pesi del modello durante l'addestramento.
    def configure_optimizers(self):
        #creazione dell'ottimizzazione
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        return optimizer

    #calcolo della perdita (loss) confrontando le predizioni del modello (y_hat) con le etichette reali (y)
    def compute_loss(self, y_hat, y):
        return self.loss_function.forward(y_hat, y)
