# Seafloor Debris Detection using Integrated ResUNet and Yolov8 Model

## Overview

This project presents a deep learning-based framework for detecting seafloor debris from underwater images. The proposed approach first restores degraded underwater images using **ResUNet** and then detects marine debris using **YOLOv8**. By enhancing image quality before detection, the system improves the visibility of underwater objects and increases detection accuracy.

## Objectives

* Enhance degraded underwater images using ResUNet.
* Detect and localize seafloor debris using YOLOv8.
* Improve detection performance in challenging underwater environments.
* Support marine pollution monitoring and ocean conservation.

## Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Image Processing:** OpenCV, NumPy
* **Object Detection:** YOLOv8
* **Image Restoration:** ResUNet
* **Development Environment:** Jupyter Notebook

## Dataset

### EUVP Dataset

* Used to train the ResUNet image restoration model.
* Clean underwater images are used as ground truth.
* Artificially degraded images are generated for supervised training.

### TrashCan Dataset

* Used to train the YOLOv8 object detection model.
* Contains annotated underwater images with different categories of marine debris.

## Methodology

1. Collect underwater images from the EUVP and TrashCan datasets.
2. Generate degraded images using:

   * Color attenuation
   * Contrast reduction
   * Gaussian blur
   * Backscatter (underwater haze)
   * Gaussian noise
3. Train the ResUNet model to restore degraded underwater images.
4. Use the restored images as input to the YOLOv8 model.
5. Detect and localize seafloor debris using bounding boxes.

## Model Architecture

### Image Restoration (ResUNet)

* 4 Encoder Residual Blocks
* 1 Bottleneck Residual Block
* 4 Decoder Residual Blocks
* Final 1 × 1 Convolution Layer

### Object Detection

* YOLOv8 Object Detection Model

## Features

* Underwater image enhancement
* Noise and haze reduction
* Contrast and color restoration
* Accurate debris detection
* End-to-end deep learning pipeline
