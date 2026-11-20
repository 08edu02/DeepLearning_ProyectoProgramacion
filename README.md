# Comparación de Estrategias para la Clasificación de Imágenes: CNN *From Scratch* vs. *Transfer Learning*

**Maestría:** Maestría en Inteligencia Artificial  
**Asignatura:** 07MIAR_04_A_2026-27 - Redes Neuronales y Deep Learning  
**Actividad:** Proyecto de programación C1 — *Deep Vision in classification tasks*  

### Integrantes
- Carlos Eduardo Chavarria Mujica — carloseduardo7366@gmail.com — [GitHub](https://github.com/08edu02)
- Everth Roman Palomino Lanchipa — epalominola@gmail.com — [GitHub](https://github.com/epalominola)
- Luis Alberto Chumpitaz Santivañez — l.chumpitaz.s@outlook.com — [GitHub](https://github.com/uluisgit)

---

## 1. Descripción del Proyecto

Este proyecto aborda de forma experimental un problema de visión por computador enfocado en la **clasificación supervisada multiclase de imágenes** en el dominio agroalimentario.

El objetivo principal es evaluar, comparar y justificar empíricamente el rendimiento, la convergencia y la capacidad de generalización entre dos enfoques fundamentales del Deep Learning:
1. **Entrenamiento desde cero (*From Scratch*):** Diseño iterativo de una arquitectura convolucional (CNN) propia, partiendo de un *Baseline* sin regularizar hasta un modelo optimizado con *Data Augmentation*, *Batch Normalization*, regularización L2, *Dropout* y *GlobalAveragePooling2D*.
2. **Aprendizaje por transferencia (*Transfer Learning* y *Fine-Tuning*):** Comparativa entre **MobileNetV2** y **ResNet50** preentrenadas en *ImageNet*, seleccionando la arquitectura óptima por parsimonia y eficiencia para aplicar un ajuste fino (*Fine-Tuning*) sobre el 25% superior de sus capas.

El flujo de trabajo cubre el ciclo completo: descarga automatizada, auditoría y análisis exploratorio (EDA), acondicionamiento con `tf.data`, experimentación progresiva y evaluación sobre un conjunto de prueba no visto.

---

## 2. Conjunto de Datos

Se utiliza el **Vegetable Image Dataset**, obtenido desde Kaggle:
- **Fuente:** [Vegetable Image Dataset (Kaggle)](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Número de clases:** 15 categorías botánicas de hortalizas.
- **Volumen:** 21.000 imágenes RGB en formato JPG (0 archivos corruptos).
- **Resolución base:** $224 \times 224$ píxeles (con redimensionamiento explícito en el *pipeline* tras detectar 9 imágenes con alturas atípicas en el EDA).
- **Distribución:** Perfectamente balanceada (1.400 imágenes por clase: 1.000 para entrenamiento, 200 para validación y 200 para prueba).

> **Cita bibliográfica:**  
> Ahmed, M. I., Mamun, S. M., & Asif, A. U. Z. (2021). *DCNN-based vegetable image classification using transfer learning: A comparative study*. 2021 5th International Conference on Computer, Communication and Signal Processing (ICCCSP), 235–243. IEEE.

---

## 3. Resultados Principales (Conjunto de Test)

| Estrategia / Modelo | Parámetros | Test Accuracy | Test Loss | Observaciones |
| :--- | :---: | :---: | :---: | :--- |
| **1.1 CNN *From Scratch* (Baseline)** | 12,94 M | 95.07% | 0.2459 | Sobreajuste evidente por uso de `Flatten` sin regularización. |
| **1.2 CNN *From Scratch* (Regularizada)** | 490 K | 96.77% | 0.2562 | Reducción del 96.2% de parámetros y mejora de +1.70 pp. |
| **2. *Transfer Learning* (ResNet50)** | >23,5 M | 99.97% | 0.0252 | Empate técnico en validación, mayor coste computacional. |
| **2. *Transfer Learning* (MobileNetV2)** | 2,26 M | 99.87% | 0.0224 | **Seleccionada por eficiencia y parsimonia** (4 errores de 3.000). |
| **3. *Fine-Tuning* (MobileNetV2 - Top 25%)** | 2,26 M | 99.90% | 0.0182 | Efecto techo: mejora marginal de 1 imagen (3 errores de 3.000). |

**Conclusión clave:** *MobileNetV2* con *Transfer Learning* puro ofrece el mejor equilibrio entre precisión (~99.9%), tamaño de modelo y velocidad de inferencia para un entorno de producción.

---

## 4. Estructura del Repositorio

Los datos binarios pesados están excluidos del control de versiones mediante `.gitignore`:

```text
DeepLearning_ProyectoProgramacion/
├── data/                                        # Dataset local descargado automáticamente (ignorado en Git)
├── .gitignore                                   # Reglas de exclusión de Git
├── README.md                                    # Documentación del proyecto
├── requirements.txt                             # Dependencias y versiones utilizadas
├── Grupo05_Chavarria_Palomino_Chumpitaz.ipynb   # Notebook principal reproducible con todo el pipeline
├── Grupo05_Chavarria_Palomino_Chumpitaz.html    # Exportación interactiva del notebook en HTML
└── Grupo05_Chavarria_Palomino_Chumpitaz.pdf     # Memoria y resultados exportados en PDF
```

---

## 5. Instalación y Ejecución

### Entorno de ejecución probado

* **Hardware:** NVIDIA GeForce RTX 3090 (con política `mixed_float16` activada), 32 GB RAM.
* **Software:** WSL2 + Miniconda, Python 3.12.16, TensorFlow 2.21.0.
* *El notebook es totalmente compatible tanto con ejecución local como con **Google Colab** (detección automática de entorno).*

### 1. Clonar el repositorio

```bash
git clone [https://github.com/08edu02/DeepLearning_ProyectoProgramacion](https://github.com/08edu02/DeepLearning_ProyectoProgramacion)
cd DeepLearning_ProyectoProgramacion
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecución del Notebook

Abre y ejecuta las celdas del archivo `.ipynb`.

* **Nota sobre los datos:** La descarga desde Kaggle y la extracción en `data/` son **completamente automáticas** en las primeras celdas del notebook, sin requerir intervención manual previa.