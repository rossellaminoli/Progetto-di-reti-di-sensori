#classe del mio thingy

#cosa ci deve essere nella relazione:
#introduzione che dice cosa voglio riconoscere
#metodologia -> parte che descrive com'è fatta la nostra applicazione, descrizione del codice, immaginare di raccontarlo a una perosna che poi possa replicarlo
#discussione dei risultati -> riportiamo esito, matrice di confusione e risultato migliore che abbiamo ottenuto e li commentiamo in senso critico
#
#posso fare 3 main, nella mia classe ho tutto quello che mi serve per collezionare dati, fare il training e fare l'interferenza

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from bleak import BleakClient, BLEDevice
from utils.UUIDs import TMS_CONF_UUID,TMS_RAW_DATA_UUID
import struct
from datetime import datetime
from functools import partial
from utils.utility import motion_characteristics, change_status, scan, find
#from utils.utility import get_uuid
import onnxruntime as ort
import numpy as np


#creo una mia classe e la faccio derivare da BleakClient, prendo tutte le cose di BleakClient e ci aggiungo quelle che mi servono
#il self mi indica la classe stessa, se faccio self. accedo a dei valori/metodi della classe
#super chiama la classe superiore cioè BleakClient
#quando modifico un metodo della classe padre si dice che sto facendo un overhead della classe padre

#devo aggiungere il modello

#dentro la classe ho 3 parti:
#-parte relazionata con il sensore -> di ricezione dei dati inerziali(receive_inertial_data)
#-salvataggio dati
#-fase d'inferenza, mi calcolo un modello per inferire i dati

#todo da fare: andare nel main e specificare cosa vogliamo fare, se ricevere solo dati o fare anche una classificazione:
#inserisco path_to_model: str=None e poi dico se path=None non vado a istanziare nessun modello

class Thingy52Client(BleakClient):

    def __init__(self, device: BLEDevice): #il parametro device deve essere di tipo BLEDevice
        super().__init__(device)
        self.mac_address = device.address

        #self.model = ort.InferenceSession('training/CNN_60.onnx')
        #self.classes = ["cycling", "skipping", "standing"]

        self.buffer_size = 1_000
        self.data_buffer = {"x": [], "y": [], "z": []}

        """
        # Data buffer
        self.buffer_size = 6
        self.data_buffer = []
        """

        #Recording information
        self.recording_name=None #nome del file in cui registro i dati
        self.file=None #puntatore a un file aperto, mando il file alla open e la open gli assegna un numero

    async def connect(self, **kwargs) -> bool:
        """
        Connect to the Thingy52 device
        :return: True if the connection is successful, False otherwise
        """

        print(f"Connecting to {self.mac_address}")
        await super().connect(**kwargs)

        try:
            print(f"Connected to {self.mac_address}")
            await change_status(self, "connected")
            return True
        except Exception as e: #cattura tutte le eccezioni e le salva nella variabile e
            print(f"Failed to connect to {self.mac_address}")
            return False


    async def disconnect(self) -> bool:
        """
        Disconnect from the Thingy52 device
        :return: True if the disconnection is successful, False otherwise
        """
        print(f"Disconnecting from {self.mac_address}")

        #await self.stop_notify(TMS_RAW_DATA_UUID)
        await self.file.close()
        await super().disconnect()

        try:
            print(f"Disconnected from {self.mac_address}")
            return True
        except Exception as e:
            print(f"Failed to disconnect from {self.mac_address}")
            return False




    async def receive_inertial_data(self, sampling_frequency: int = 60):
        """
        Receive data from the Thingy52 device
        :return: None
        """

        # Set the sampling frequency
        payload = motion_characteristics(motion_processing_unit_freq=sampling_frequency)
        await self.write_gatt_char(TMS_CONF_UUID, payload)

        #Open the file to save data
        self.file=open(self.recording_name, "a+") #a+ -> append, apro il file in modalità append cioè aggiungo dei dati nel file

        # Ask to activate the raw data (Inertial data)
        await self.start_notify(TMS_RAW_DATA_UUID, self.raw_data_callback)

        # Change the LED color to red, recording status
        await change_status(self, "recording")


        try:
            while True:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            await self.stop_notify(TMS_RAW_DATA_UUID)
            print("Stopped notification")

    def save_to(self,file_name):
        self.recording_name = f"{self.mac_address.replace(':', '-')}_{file_name}.csv"

    # Callbacks
    def raw_data_callback(self, sender, data):

        # Handle the incoming accelerometer data here
        receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Accelerometer
        acc_x = (struct.unpack('h', data[0:2])[0] * 1.0) / 2 ** 10
        acc_y = (struct.unpack('h', data[2:4])[0] * 1.0) / 2 ** 10
        acc_z = (struct.unpack('h', data[4:6])[0] * 1.0) / 2 ** 10

        # Gyroscope
        gyro_x = (struct.unpack('h', data[6:8])[0] * 1.0) / 2 ** 5
        gyro_y = (struct.unpack('h', data[8:10])[0] * 1.0) / 2 ** 5
        gyro_z = (struct.unpack('h', data[10:12])[0] * 1.0) / 2 ** 5

        # Compass
        comp_x = (struct.unpack('h', data[12:14])[0] * 1.0) / 2 ** 4
        comp_y = (struct.unpack('h', data[14:16])[0] * 1.0) / 2 ** 4
        comp_z = (struct.unpack('h', data[16:18])[0] * 1.0) / 2 ** 4

        # Save the data to a file
        self.file.write(f"{receive_time}, {acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z}\n")#devo chiudere il file sennò spreco risorse


        # Update the data buffer
        # Controlla se la lunghezza del buffer per x supera o è uguale a buffer_size.
        # Se sì, rimuove(con pop) il primo elemento (più vecchio) dalle liste x, y e z per fare spazio ai nuovi dati.

        if len(self.data_buffer["x"]) >= self.buffer_size:
            self.data_buffer["x"].pop(0)
            self.data_buffer["y"].pop(0)
            self.data_buffer["z"].pop(0)
        self.data_buffer["x"].append(acc_x)#Aggiunge i nuovi dati dell'accelerometro (acc_x, acc_y, acc_z) alle liste corrispondenti nel buffer.
        self.data_buffer["y"].append(acc_y)
        self.data_buffer["z"].append(acc_z)

        #Utilizza una formatted string(f - string) per stampare i dati dell'accelerometro in un formato leggibile.
        print(f"\r{self.mac_address} | {receive_time} - Accelerometer: X={acc_x: 2.3f}, Y={acc_y: 2.3f}, Z={acc_z: 2.3f}",
            end="", flush=True)

        """
        # Update the data buffer
        if len(self.data_buffer) == self.buffer_size:
            input_data = np.array(self.data_buffer, dtype=np.float32).reshape(1, self.buffer_size, 6)
            input_ = self.model.get_inputs()[0].name
            cls_index = np.argmax(self.model.run(None, {input_: input_data})[0], axis=1)[0]
            print(f"\r{self.mac_address} | {receive_time} - Prediction: {self.classes[cls_index]}", end="", flush=True)
            self.data_buffer.clear()

        self.data_buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        """
