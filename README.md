# Image Inpainting using Deep Learning with U-Net on CIFAR-10

This project demonstrates the use of deep learning techniques, specifically the U-Net model, for image inpainting on the CIFAR-10 dataset. The task involves restoring missing parts of an image, with the model learning to predict and reconstruct the missing regions based on the surrounding context.

### Project Thesis:
This project is part of a larger thesis that investigates the use of deep learning models for image inpainting. The U-Net architecture, originally designed for image segmentation, is adapted to handle image inpainting tasks.

## Project Overview

- **Dataset**: CIFAR-10, which contains 60,000 32x32 color images in 10 classes.
- **Model Architecture**: U-Net, adapted for the inpainting task.
- **Task**: Image inpainting, where random portions of images are removed, and the model is trained to predict the missing regions.

The project is implemented in a Jupyter Notebook (`inpainting_unet.ipynb`), which includes both the code and the project thesis section.

## Requirements

To run the notebook and train the model, the following libraries are required:
- Python 3.x
- TensorFlow/Keras
- Numpy
- Matplotlib
- OpenCV
- CIFAR-10 dataset

## Setup Instructions
 Clone the repository:
    ```bash
    git clone https://github.com/your-username/image-inpainting-unet.git
   
    ```

## Running the Model

The model and training process are implemented in the `inpainting_unet.ipynb` notebook. It covers the following steps:
1. **Preprocessing the CIFAR-10 dataset**: Loading and augmenting images for the inpainting task.
2. **Model Training**: Training the U-Net model for image inpainting.
3. **Evaluation**: Evaluating the model performance on unseen test data.
4. **Results**: Visualizing the inpainted images and comparing them with the original images.

## Model Architecture

The U-Net model used in this project consists of the following components:
1. **Encoder (Contracting Path)**: A series of Conv2D layers with ReLU activations, followed by MaxPooling2D for downsampling.
2. **Bottleneck**: Feature extraction layers.
3. **Decoder (Expanding Path)**: UpSampling2D layers to restore the image dimensions, concatenated with the encoder features.
4. **Output Layer**: A final Conv2D layer with a `sigmoid` activation for pixel-wise inpainting.


## Thesis Section

The notebook includes a section that explains the theoretical background and methodology behind the image inpainting task. This section is part of the broader research on deep learning applications in image reconstruction and generation.


## Acknowledgements

- The U-Net architecture is inspired by the original work on U-Net for image segmentation.
- CIFAR-10 dataset is widely used for machine learning and computer vision tasks.

