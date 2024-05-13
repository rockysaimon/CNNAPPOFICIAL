import eel
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import pefile
import numpy as np
import base64
import json

# Inicializar Eel
eel.init('web')

class MalwarePredictor:
    def __init__(self):
        # Cargar el modelo de CNN previamente entrenado
        self.model = tf.keras.models.load_model(r'train_model\modelo3.keras')

    @staticmethod
    def convert_to_binary_image(file_path):
        try:
            pe = pefile.PE(file_path)
            sections = ['.data', '.rsrc', '.rdata', '.text']
            binary_image = np.zeros((256, 256), dtype=np.uint8)  # Usar solo dos dimensiones ya que la imagen es en escala de grises

            for section in pe.sections:
                section_name = section.Name.decode().strip('\x00')
                if section_name in sections:
                    start = section.PointerToRawData
                    end = start + section.SizeOfRawData
                    data = section.get_data()
                    section_image = np.frombuffer(data, dtype=np.uint8).reshape(-1, 256)[:256, :]

                    # Ajustar la forma de section_image si es necesario
                    if section_image.shape[0] < 256:
                        pad_size = 256 - section_image.shape[0]
                        section_image = np.pad(section_image, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

                    # Copiar la sección en la posición adecuada de binary_image
                    binary_image = np.maximum(binary_image, section_image)

            # Convertir la imagen a una imagen RGB repitiendo el canal de color
            rgb_image = np.repeat(np.expand_dims(binary_image, axis=-1), 3, axis=-1)
            return rgb_image
        except Exception as e:
            raise Exception(f'Error al convertir el archivo a imagen binaria: {str(e)}')


    @staticmethod
    def predict_malware(binary_image):
        try:
            predictions = malware_predictor.model.predict(np.expand_dims(binary_image, axis=0))
            predicted_class = np.argmax(predictions)
            probability = predictions[0][predicted_class]

            return predicted_class, probability
        except Exception as e:
            raise Exception(f'Error al realizar la predicción: {str(e)}')

        
    import numpy as np

    @eel.expose
    def showPrediction(self, class_name, probability):
        # Mostrar la predicción en consola
        classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.gen!g', 'C2LOP.P', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J','No Malware', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
        clase=classes[class_name]
        try:
            # Convertir class_name y probability a tipos de datos primitivos de Python
            clase = clase.item() if isinstance(clase, np.generic) else clase
            probability = probability.item() if isinstance(probability, np.generic) else probability

            print(clase, probability)
            if clase is not None and probability is not None:
                eel.showPrediction(json.dumps(clase), json.dumps(probability))
        except KeyError as e:
            print(f'Error al procesar el mensaje: {e}') 



    @staticmethod
    @eel.expose
    def select_file():
        try:
            app = QApplication([])  # Crear la aplicación
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(None, 'Seleccionar archivo', '', 'Executable files (*.exe)', options=options)
            if file_path:
                binary_image = MalwarePredictor.convert_to_binary_image(file_path)
                class_name, probability = MalwarePredictor.predict_malware(binary_image)
                
                # Convertir la imagen a base64 para enviarla a JavaScript
                with open('binary_image.png', 'wb') as f:
                    plt.imsave(f, binary_image.squeeze(), cmap='gray', format='png')
                with open('binary_image.png', 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                # Mostrar la predicción en la interfaz gráfica
                malware_predictor.showPrediction(class_name, probability)  # Aquí usamos la instancia de MalwarePredictor
                eel.showImage(image_data)
        except KeyError as e:
            print(f'Error al procesar el mensaje: {e}')

# Crear una instancia de la clase MalwarePredictor
malware_predictor = MalwarePredictor()

# Ejecutar la aplicación
eel.start('index.html', size=(max, max), port=8080)
