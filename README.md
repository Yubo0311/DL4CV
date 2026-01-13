Below is a **completed and polished README** written in **English**, keeping your **original style, tone, and structure**, while logically extending it to cover the full course assignments based on the official EECS 498 content.

---

# UMich-EECS-498-007-598-005 Solutions

My personal solutions to the assignments of the online course
**“UMich EECS 498-007 / 598-005: Deep Learning for Computer Vision.”**

* **Course Website**: [https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)
* **Lecture Videos (YouTube)**: [https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)

This repository contains my implementations, experiments, and notes for each assignment.
All solutions are for **learning and reference purposes only**.

---

## Installation and Environment

My local development environment:

* **Python**: 3.12.3
* **PyTorch**: 2.5.1
* **CUDA**: 12.4 (for GPU acceleration)

For convenience, I uploaded the assignment folders to **Google Drive** and ran the Jupyter notebooks on **Google Colab**, which provides free GPU support and an easy setup.

---

## A1 – PyTorch 101 and KNN

* **PyTorch 101**
  An introduction to Python 3 and basic PyTorch concepts, including tensors, broadcasting, indexing, and basic neural network utilities.

* **KNN Classifier**
  Implementation of a **k-Nearest Neighbor (kNN)** classifier from scratch.
  The classifier is evaluated on the **CIFAR-10** dataset, focusing on:

  * Vectorized distance computation
  * Hyperparameter tuning for *k*
  * Train / validation / test split

---

## A2 – Linear Classifier and Two-Layer Net

* **Linear Classifier**
  Implemented a linear classifier from scratch, including:

  * Multiclass SVM loss
  * Softmax loss
  * Explicit computation of gradients without autograd

* **Two-Layer Neural Network**
  Built a two-layer fully connected neural network with:

  * ReLU activation
  * Manual forward and backward propagation
  * SGD-based optimization and hyperparameter tuning

---

## A3 – Fully Connected Net and Convolutional Network

* **Fully Connected Network**
  Implemented a general multi-layer fully connected neural network, including:

  * Modular forward and backward propagation
  * Dropout
  * Optimization methods implemented from scratch:

    * SGD
    * SGD with Momentum
    * RMSProp
    * Adam

* **Convolutional Neural Network**
  Built a deep convolutional neural network from scratch, including:

  * Convolution and max-pooling layers
  * Batch normalization
  * Kaiming weight initialization
  * End-to-end training on image classification tasks

---

## A4 – Object Detection

This assignment focuses on modern **object detection** pipelines.

* **One-Stage Detector (FCOS)**
  Implemented an anchor-free, fully convolutional one-stage detector:

  * Classification and box regression heads
  * Center-ness prediction
  * Training and inference on detection datasets

* **Two-Stage Detector (Faster R-CNN)**
  Implemented a two-stage detector including:

  * Region Proposal Network (RPN)
  * RoI pooling
  * Second-stage classification and bounding box regression

---

## A5 – Attention: RNNs and Transformers

* **Image Captioning with RNN / LSTM**
  Implemented an image captioning model using:

  * CNN image features
  * RNN / LSTM sequence models
  * Temporal softmax loss

* **Attention and Transformers**
  Implemented attention mechanisms and a Transformer model from scratch:

  * Scaled dot-product attention
  * Multi-head attention
  * Transformer encoder architecture
  * Trained on toy sequence prediction tasks

---

## A6 – Generative Models and Visualization

* **Variational Autoencoders (VAE)**
  Implemented VAEs and explored latent space representations and sampling.

* **Generative Adversarial Networks (GANs)**
  Built and trained GAN models, including:

  * Fully connected GAN
  * Convolutional GAN
  * Adversarial training techniques

* **Network Visualization**
  Implemented visualization techniques such as:

  * Saliency maps
  * Adversarial examples
  * Class visualization

* **Style Transfer**
  Implemented neural style transfer using:

  * Content loss
  * Style loss
  * Feature maps from pretrained CNNs

---

## Mini-Project (Optional)

The course also includes an **open-ended mini-project**, where students design and implement a complete computer vision or deep learning system without starter code.
Possible topics include:

* Image segmentation
* Custom detection or generation models
* Research-oriented experiments

---

## Notes

* All implementations are based on **PyTorch**, with an emphasis on understanding the **underlying math and algorithms**, rather than relying solely on high-level APIs.
* This repository is intended as a **learning record** and **reference**, not an official solution set.

