# NLLB Model Deployment and TensorRT Inference Acceleration

This folder contains the code and configuration for deploying the **NLLB (No Language Left Behind)** translation model and accelerating inference using **TensorRT**.

## 📘 Project Overview
NLLB is a multilingual machine translation model developed by Meta AI, designed to provide high-quality translation across over 200 languages.  
This project focuses on the following features:
- Deploying and loading the model locally;
- Optimizing inference performance with TensorRT acceleration;
- Comparing inference performance and memory usage.

## ⚙️ Folder Structure

├── infer.py # Main inference script
├── onnx_to_trt.py # Script for converting ONNX model to TensorRT engine
├── test.py # Basic test script for model inference
├── torch_to_onnx.py # Script for converting PyTorch model to ONNX format
├── onnx_infer2.py # Additional ONNX inference script
├── onnx_to_trt2.py # Alternative TensorRT conversion script
├── test2.py # Additional test script for model inference
└── trt_infer.py # TensorRT inference script


## 🚀 Features
- Support for exporting the original NLLB model weights to ONNX format;
- Converting ONNX models to TensorRT engines for faster inference;
- Performance comparison (native inference vs TensorRT acceleration);
- Scalable for multi-GPU or multi-language batch translation tasks.

## 🔗 Original Repository
This project is based on the official Meta NLLB model. You can find the original repository here:  
👉 [https://github.com/facebookresearch/fairseq/tree/nllb](https://github.com/facebookresearch/fairseq/tree/nllb)

## 📄 References
- [NLLB Official Paper: No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)
- [TensorRT Official Documentation](https://developer.nvidia.com/tensorrt)

---

> 🧩 This project is for research and educational purposes only. All copyrights belong to the original authors and respective organizations.
