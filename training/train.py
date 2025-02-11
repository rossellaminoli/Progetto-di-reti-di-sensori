import torch
from torch.utils.data import DataLoader, Subset
#datLoader è una classe che permette di caricare il dataset in mini-batch durante l'addestramento
#subset è una classe per creare un sottoinsieme di un dataset
from sklearn.model_selection import KFold
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from CNN import CNN
from dataset import CSVDataModule

torch.set_float32_matmul_precision('medium')
#Questo comando configura PyTorch per eseguire operazioni di moltiplicazione di matrici con precisione medio-alta quando si utilizzano tensori di tipo float32.
torch.backends.cudnn.benchmark = True
#È un flag che, quando impostato a True, permette a cuDNN di eseguire un benchmarking delle diverse implementazioni di convoluzione e scegliere quella ottimale per il tuo hardware.

#in checkpoint -> bestcheckpoint contiene il migliore modello -> posso generare un file di tipo onnx
#nel main seleziono il fold con il miglior modelllo


def main():
    path = "data" #percorso della directory dove si trova il dataset, dove si trovano i miei file
    k = 5 #numero di fold per la validazione incrociata K-fold
    window_size = 1 #Dimensione della finestra di tempo (in secondi) utilizzata per il preprocessing dei dati
    #creazione del modulo dei dati -> modulo personalizzato per gestire caricamento dati da file csv
    sampling_frequency=60#Hz
    data_module = CSVDataModule(
        root_dir=path,#crea i vari fold con gli indici relativi e crea anche le finestre di dati
        batch_size=32,#parametro che va a definire quanti dati gli diamo in input a ogni step del train prima che questo faccia l'update del train, dipende dalla quantità di dati che abbiamo
        k_folds=k,
        window_size=window_size, # secondi
    )

    data_module.setup() #configura e prepara i dati per l'addestramento

    # Perform k-fold cross-validation -> nel dataset ho p1,p2,p3 e ogni volta faccio il training su 2 e il test sull'altro, alla fine del ciclo testo tutti e tre
    #o prendo il modello che funziona meglio o faccio il model sampling(modello che li unisce tutti e poi decide in base alla maggioranza)
    for fold in range(data_module.k):
        print(f'Fold {fold + 1}/{data_module.k + 1}')

        #per ogni fold istanzio un modello
        model = CNN(window_size * sampling_frequency, fold + 1, classes_names=data_module.legend)

        # Define checkpoint callback to save the best model for each fold
        #salva il modello migliore all'interno del ciclo di train sul validation loss -> perdita dei dati
        #più la loss è piccola e più il modello va bene, voglio il modello con la loss più bassa
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/fold_{fold}',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        #esegue training del modello
        trainer = Trainer(
            max_epochs=20,
            accelerator="cpu", #devo mettere cpu se non ho la gpu
            devices=1,
            logger=True,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback]
        )

        # Train the model -> uso fit per iniziare l'addestramento
        trainer.fit(model, data_module.train_dataloader(fold=fold), data_module.val_dataloader(fold=fold))

        # Test the model -> testo il modello sui miei dati di valutazione
        trainer.test(model, data_module.val_dataloader(fold=fold))


if __name__ == '__main__':
    main()