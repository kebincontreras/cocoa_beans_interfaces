import gradio as gr
import cv2
import torch
from PIL import Image
import numpy as np

# Cargar el modelo YOLO (usando YOLOv5 como ejemplo)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Puedes cambiar 'yolov5s' por cualquier otro modelo

# Función para realizar detección de objetos
def detect_objects(image):
    # Convertir la imagen a un formato compatible con OpenCV
    image = np.array(image)
    
    # Hacer la detección con YOLO
    results = model(image)
    
    # Renderizar los resultados (dibujar las cajas de detección)
    results_image = results.render()[0]
    
    return Image.fromarray(results_image)

# Interfaz de Gradio para cargar una imagen
def gradio_interface():
    with gr.Blocks() as demo:
        # Título centrado
        gr.Markdown(
            """
            <center>
            <h2>Fermentation Level Classification for Cocoa Beans</h2>
            </center>
            """
        )

        # Botón GitHub centrado justo debajo del título
        gr.Markdown(
            """
            <center>
            <a href="https://github.com/kebincontreras/cocoa_beans_interfaces" target="_blank" style="text-decoration: none;">
            <button style="background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px;">GitHub</button>
            </a>
            </center>
            """
        )

        # Organizar imágenes en la misma fila
        with gr.Row():
            img_input = gr.Image(label="Upload Image")
            img_output = gr.Image(label="Image with Detected Objects")

        # Botón para aplicar la detección de objetos a la imagen subida
        btn_detect_upload = gr.Button("Classify Fermentation Level")

        # Conectar el botón con la función de detección de objetos
        btn_detect_upload.click(detect_objects, inputs=img_input, outputs=img_output)

    return demo

# Ejecutar la aplicación
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()

