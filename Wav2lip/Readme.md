Wav2Lip Acceleration Project (TensorRT Optimization) This repository documents my work on implementing and optimizing the Wav2Lip deep learning model.

The project was executed in two distinct phases:

Architectural Implementation (From Scratch) Objective: To gain a comprehensive understanding of the Wav2Lip architecture, which ensures highly realistic lip synchronization in synthesized video.
Key Work: Implementing the full model structure from the ground up, with a focus on core components such as the Visual Quality Discriminator and the unique Lip Sync Loss function. This phase emphasized clarity in data flow and component integration.

High-Performance Optimization with TensorRT Objective: To significantly reduce infesrence latency and prepare the model for high-throughput deployment.
Key Work: Conversion of the fully implemented Wav2Lip model into an NVIDIA TensorRT (TRT) engine. This involved model graph optimization and quantization techniques to leverage GPU acceleration, resulting in substantial speed-up compared to standard framework inference.

This project showcases expertise in both neural network architecture implementation and deployment-level performance optimization.


## 🔗 Original Repository
This project is based on the official Wav2Lip repository. You can find the original repository here:  
👉 [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

## 📄 References
- [Wav2Lip: Accurately Lip-syncing Videos In The Wild](https://arxiv.org/abs/2008.10010)
- [Sync Labs: High-Definition Lip Syncing API](https://sync.so)

---

> 🧩 This project is for research and educational purposes only. All copyrights belong to the original authors and respective organizations.
