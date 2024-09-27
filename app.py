import gradio as gr
from Scripts.image_processing import apply_blur, clip_image, wrap_image
from Scripts.detection import yolov10_inference, calculate_detection_metrics
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from Scripts.utils import *
from Scripts.utils import modulo
from Scripts.utils_spud import *  # Asegúrate de que las funciones necesarias estén importadas aquí

import cv2
import matplotlib.pyplot as plt
import packaging

IMAGE_SIZE = 640  # Asignar el valor constante para image_size

def process_image(image, model_id, sat_factor, selected_method):
    conf_threshold = 0.85
    correction = 1.0
    kernel_size = 7
    
    DO = 1
    t = 0.7
    vertical = True
    
    original_image = np.array(image)
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image * 255.0
    original_image = original_image.astype(np.uint8)

    scaling = 1.0
    original_image = cv2.resize(original_image, (0, 0), fx=scaling, fy=scaling)

    blurred_image = apply_blur(original_image / 255.0, kernel_size)
    clipped_image = clip_image(blurred_image, correction, sat_factor)

    img_tensor = torch.tensor(blurred_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    wrapped_image = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    wrapped_image = (wrapped_image*255).astype(np.uint8)

    original_annotated, original_detections = yolov10_inference(original_image, model_id, IMAGE_SIZE, conf_threshold)
    clipped_annotated, clipped_detections = yolov10_inference((clipped_image*255.0).astype(np.uint8), "yolov10n", IMAGE_SIZE, conf_threshold)
    wrapped_annotated, wrapped_detections = yolov10_inference(wrapped_image, model_id, IMAGE_SIZE, conf_threshold)

    if selected_method == "SPUD":
        # Uso de la función personalizada de SPUD
        recon_image = recons_spud(img_tensor, threshold=0.1, mx=1.0)
    else:
        # Método por defecto AHFD
        recon_image = recons(img_tensor, DO=DO, L=1.0, vertical=vertical, t=t)

    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0))
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)

    recon_annotated, recon_detections = yolov10_inference(recon_image_np, model_id, IMAGE_SIZE, conf_threshold)

    metrics_clip = calculate_detection_metrics(original_detections, clipped_detections)
    metrics_wrap = calculate_detection_metrics(original_detections, wrapped_detections)
    metrics_recons = calculate_detection_metrics(original_detections, recon_detections)

    return original_annotated, clipped_annotated, wrapped_annotated, recon_annotated, metrics_clip, metrics_wrap, metrics_recons

def app():
    image_scaler = 0.55  
    image_width = int(600 * image_scaler)  
    image_height = int(200 * image_scaler)

    with gr.Blocks(css=f"""
    .fixed-size-image img {{
        width: {image_width}px;
        height: {image_height}px;
        object-fit: cover;
    }}
    .gr-row {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .gr-column {{
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 0 !important;
        margin: 0 !important;
    }}
    #centered-title {{
        text-align: center;
    }}
    #centered-text {{
        text-align: center;
        margin-bottom: 10px;
    }}
    .custom-button {{
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: bold;
        color: white;
        border-radius: 5px;
        margin-right: 5px;  /* Reducir el margen entre botones */
        margin-bottom: 0px;
        text-decoration: none;
        text-align: center;
    }}
    .btn-grey {{
        background-color: #4b4b4b;
    }}
    .btn-red {{
        background-color: #e94e42;
    }}
    .btn-blue {{
        background-color: #007bff;
    }}
    .gr-examples img {{
        width: 200px;
        height: 200px;
    }}
    """) as demo:
        gr.Markdown("## Modulo Imaging for Computer Vision", elem_id="centered-title")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### High Dynamic Range Modulo Imaging for Robust Object Detection in Autonomous Driving", elem_id="centered-text")
                with gr.Row():
                    gr.HTML('<a href="https://openreview.net/pdf?id=2GqZFx2I7s" target="_blank" class="custom-button btn-grey">Article</a>')
                    gr.HTML('<a href="https://github.com/kebincontreras/Modulo_images.git" target="_blank" class="custom-button btn-blue">GitHub</a>')

            with gr.Column():
                gr.Markdown("### Autoregressive High-Order Finite Difference Modulo Imaging: High-Dynamic Range for Computer Vision Applications", elem_id="centered-text")
                with gr.Row():
                    gr.HTML('<a href="https://cvlai.net/aim/2024/" target="_blank" class="custom-button btn-red">Article</a>')
                    gr.HTML('<a href="https://github.com/bemc22/AHFD" target="_blank" class="custom-button btn-blue">GitHub</a>')

        image = gr.Image(type="pil", label="Upload Image", interactive=True)

        model_id = gr.Dropdown(label="Model", choices=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"], value="yolov10x")
        sat_factor = gr.Slider(label="Saturation Factor", minimum=1.0, maximum=5.0, step=0.1, value=2.0)

        selected_method = gr.Radio(
            label="Select Method",
            choices=["AHFD","SPUD"],
            value="SPUD"
        )
        
        process_button = gr.Button("Process Image")

        examples = [
            ["Add_ons/imagen1.png"],
            ["Add_ons/imagen2.png"],
            ["Add_ons/imagen3.jpg"],
            ["Add_ons/imagen4.jpg"],
            ["Add_ons/imagen5.jpg"],
            ["Add_ons/imagen6.png"]
        ]

        gr.Examples(
            examples=examples,
            inputs=[image],
            label="Choose an Example Image"
        )

        with gr.Row():
            with gr.Column():
                output_original = gr.Image(label="Ground Truth")
                output_wrap = gr.Image(label="Modulo-ADC")
            with gr.Column():
                output_clip = gr.Image(label="CCD-Saturated")
                output_recons = gr.Image(label="Modulo-ADC + Recovery")

        process_button.click(
            fn=process_image,
            inputs=[image, model_id, sat_factor, selected_method],
            outputs=[output_original, output_clip, output_wrap, output_recons]
        )

    return demo

if __name__ == "__main__":
    app().launch()
