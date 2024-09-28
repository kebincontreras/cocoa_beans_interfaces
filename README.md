# cocoa_beans_interfaces
[![demo-online](https://img.shields.io/badge/demo-online-success?style=flat-square)]([https://tu-demo-url.com](https://huggingface.co/spaces/kebincontreras/Fermentation_Level_Classification_for_Cocoa_Beans))


Requisitos previos
Python 3.8 o superior.
Git para clonar el repositorio.
1. Clonar el repositorio
Primero, clona este repositorio:


git clone https://github.com/kebincontreras/cocoa_beans_interfaces.git
cd cocoa_beans_interfaces
2. Crear un entorno virtual
Para mantener las dependencias aisladas, es recomendable crear un entorno virtual.

En Linux / macOS:

python3 -m venv interfas_beans_cocoa
source interfas_beans_cocoa/bin/activate
En Windows:

python -m venv interfas_beans_cocoa
interfas_beans_cocoa\Scripts\activate
3. Instalar dependencias
Con el entorno virtual activado, instala las dependencias necesarias para el proyecto:


pip install -r requirements.txt
Este comando instalará todas las librerías especificadas en el archivo requirements.txt, que incluye Gradio, Torch, OpenCV y otras necesarias para la clasificación.

4. Ejecutar la aplicación
Con el entorno configurado y las dependencias instaladas, ejecuta la aplicación:


python app.py
Esto iniciará un servidor de Gradio que mostrará la interfaz para subir una imagen de granos de cacao y realizar la clasificación de niveles de fermentación.


