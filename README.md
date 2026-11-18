# Comparación de Estrategias para la Clasificación de Imágenes: CNN *From Scratch* vs. *Transfer Learning*

**Maestría:** Maestría en Inteligencia Artificial  
**Asignatura:** 07MIAR_04_A_2026-27 - Redes Neuronales y Deep Learning  
**Actividad:** Proyecto de programación C1 — *Deep Vision in classification tasks*  

### Integrantes
* Carlos Eduardo Chavarria Mujica - carloseduardo7366@gmail.com - https://github.com/08edu02
* Everth Roman Palomino Lanchipa - 
* Luis Alberto Chumpitaz Santivañez - - https://github.com/uluisgit

---

## 1. Descripción del Proyecto

Este proyecto aborda de forma experimental un problema de visión por computador enfocado en la **clasificación supervisada multiclase de imágenes**. 

El objetivo principal es evaluar, comparar y justificar empíricamente el rendimiento, convergencia y capacidad de generalización entre dos enfoques fundamentales del Deep Learning:
1. **Entrenamiento desde cero (*From Scratch*):** Diseño, regularización y optimización de una arquitectura convolucional (CNN) propia.
2. **Aprendizaje por transferencia (*Transfer Learning* y *Fine-Tuning*):** Adaptación de arquitecturas convolucionales preentrenadas sobre el conjunto *ImageNet* (evaluando al menos dos modelos como ResNet, MobileNetV2 o VGG).

El flujo de trabajo cubre el ciclo completo: ingestión automatizada, análisis exploratorio (EDA), acondicionamiento de datos con `tf.data`, experimentación progresiva de modelos y evaluación sobre datos de prueba no vistos.

---

## 2. Conjunto de Datos

Se utiliza el **Vegetable Image Dataset**, obtenido desde la plataforma Kaggle:
* **Fuente:** [Vegetable Image Dataset (Kaggle)](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
* **Número de clases:** 15 categorías botánicas de hortalizas.
* **Volumen:** 21.000 imágenes RGB en formato JPG.
* **Resolución base:** $224 \times 224$ píxeles.
* **Distribución:** Balanceada (1.400 imágenes por clase: 1.000 para entrenamiento, 200 para validación y 200 para prueba).

> **Cita bibliográfica:**  
> Ahmed, M. I., Mamun, S. M., & Asif, A. U. Z. (2021). *DCNN-based vegetable image classification using transfer learning: A comparative study*. 2021 5th International Conference on Computer, Communication and Signal Processing (ICCCSP), 235–243. IEEE.

---

## 3. Estructura del Repositorio

Los datos binarios pesados y los pesos de los modelos están explícitamente excluidos del control de versiones (`.gitignore`):

```text
DeepLearning_ProyectoProgramacion/
├── data/                  # Directorio local para almacenamiento del dataset (ignorado)
├── logs/                  # Registros de entrenamiento y eventos de TensorBoard (ignorado)
├── models/                # Checkpoints y pesos serializados (.keras, .h5) (ignorado)
├── reports/               # Exportación del notebook en formato PDF (nbconvert)
├── .gitignore             # Reglas de exclusión de Git
├── README.md              # Documentación del proyecto
├── requirements.txt       # Dependencias y versiones de librerías utilizadas
└── proyecto.ipynb         # Notebook principal reproducible con todo el pipeline ejecutado
```

---

## 4. Instalación y Ejecución

### Prerrequisitos
Se recomienda utilizar un entorno virtual con **Python 3.10 o superior** y soporte para GPU (CUDA/cuDNN o aceleración por hardware en Google Colab).

### 1. Clonar el repositorio
```bash
git clone https://github.com/08edu02/DeepLearning_ProyectoProgramacion
cd DeepLearning_ProyectoProgramacion
```

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecución del Notebook
Abre y ejecuta las celdas del archivo `proyecto.ipynb`. 
* **Nota sobre los datos:** La descarga y extracción de los datos es **completamente automática** y se ejecuta dentro de las primeras celdas del notebook sin requerir intervención manual previa en la carpeta `data/`.