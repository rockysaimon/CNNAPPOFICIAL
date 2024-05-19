import unittest
import numpy as np
import tensorflow as tf
import pefile
import base64
import json
import eel
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog

# Inicializar Eel
eel.init('web')

# Define la clase MalwarePredictor
class MalwarePredictor:
    def __init__(self, model_path):
        # Cargar el modelo de CNN previamente entrenado
        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def convert_to_binary_image(file_path):
        try:
            pe = pefile.PE(file_path)
            sections = ['.data', '.rsrc', '.rdata', '.text']
            binary_image = np.zeros((256, 256), dtype=np.uint8)

            for section in pe.sections:
                section_name = section.Name.decode().strip('\x00')
                if section_name in sections:
                    start = section.PointerToRawData
                    end = start + section.SizeOfRawData
                    data = section.get_data()
                    section_image = np.frombuffer(data, dtype=np.uint8).reshape(-1, 256)[:256, :]

                    if section_image.shape[0] < 256:
                        pad_size = 256 - section_image.shape[0]
                        section_image = np.pad(section_image, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

                    binary_image = np.maximum(binary_image, section_image)

            rgb_image = np.repeat(np.expand_dims(binary_image, axis=-1), 3, axis=-1)
            return rgb_image
        except Exception as e:
            raise Exception(f'Error al convertir el archivo a imagen binaria: {str(e)}')

    def predict_malware(self, binary_image):
        try:
            predictions = self.model.predict(np.expand_dims(binary_image, axis=0))
            predicted_class = np.argmax(predictions)
            probability = predictions[0][predicted_class]

            return predicted_class, probability
        except Exception as e:
            raise Exception(f'Error al realizar la predicción: {str(e)}')

    def showPrediction(self, class_name, probability):
        classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.gen!g', 'C2LOP.P', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J','No Malware', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']
        clase = classes[class_name]
        try:
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
                malware_predictor = MalwarePredictor(r'train_model\modelo3.keras')
                binary_image = malware_predictor.convert_to_binary_image(file_path)
                class_name, probability = malware_predictor.predict_malware(binary_image)

                # Convertir la imagen a base64 para enviarla a JavaScript
                with open('binary_image.png', 'wb') as f:
                    plt.imsave(f, binary_image.squeeze(), cmap='gray', format='png')
                with open('binary_image.png', 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

                # Mostrar la predicción en la interfaz gráfica
                malware_predictor.showPrediction(class_name, probability)
                eel.showImage(image_data)
        except KeyError as e:
            print(f'Error al procesar el mensaje: {e}')

class TestMalwarePredictor(unittest.TestCase):
    def test_convert_to_binary_image(self):
        model_path = r'train_model\modelo3.keras'  # Ruta al modelo
        malware_predictor = MalwarePredictor(model_path)

        # Prueba la conversión de archivo a imagen binaria
        file_path = r'C:\Users\zapat\Downloads\ChromeSetup.exe'  # Ruta a un archivo de prueba
        binary_image = malware_predictor.convert_to_binary_image(file_path)
        self.assertEqual(binary_image.shape, (256, 256, 3))

    def test_predict_malware(self):
        model_path = r'train_model\modelo3.keras'  # Ruta al modelo
        malware_predictor = MalwarePredictor(model_path)

        # Cargar un archivo de prueba conocido
        file_path = r"C:\Users\zapat\Downloads\ChromeSetup.exe"  # Ruta a un archivo de prueba
        binary_image = malware_predictor.convert_to_binary_image(file_path)

        # Prueba la predicción de malware
        predicted_class, probability = malware_predictor.predict_malware(binary_image)
        self.assertIsInstance(predicted_class, np.int64)
        self.assertIsInstance(probability, np.float32)

if __name__ == "__main__":
    # Ejecutar las pruebas unitarias
    unittest_result = unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestMalwarePredictor))

    # Verificar si las pruebas pasaron
    if unittest_result.wasSuccessful():
        # Iniciar la aplicación si las pruebas pasan
        eel.start('index.html', size=(max, max), port=8080)
    else:
        print("Las pruebas unitarias no pasaron. No se iniciará la aplicación.")
