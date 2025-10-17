# RAG-based PDF Query Model for Information Retrieval

This project implements an information retrieval system based on **Retrieval-Augmented Generation (RAG)**, designed to handle PDF queries and answer generation with a focus on mitigating hallucination problems in large language models (LLMs). The system is inspired by **ChatPDF** and similar models but is designed with a more efficient approach using a smaller parameter set.

## 📘 Project Overview

This project leverages **RAG** to solve the challenges faced by large language models, such as hallucinations and catastrophic forgetting. By dynamically augmenting the model with external knowledge sources, the system can retrieve relevant information and generate more accurate answers to user queries, particularly in specialized domains. The solution includes a back-end service and a front-end interface for PDF-based queries.

## ⚙️ Folder Structure

├── app.py # Main application entry point for querying PDF files
├── app2.py # Alternative application script
├── embedding.py # Script for embedding text into vectors
├── glm.py # Script for integrating with large models
├── main.py # main code for the program
├── ocr3.py # Another OCR implementation
├── pdfquery.py # Script for querying PDFs
├── requirements.txt # List of Python dependencies
├── result.png # Example result image
└── streamlitui.py # Streamlit interface script for easy deployment


## 🚀 Features

- **PDF Query Handling**: The system allows users to upload PDFs and ask questions based on their content.
- **RAG-based Query Resolution**: Combines text retrieval from a database with generative models to answer queries more accurately.
- **Efficient Model**: Achieves similar performance to state-of-the-art models while using significantly fewer parameters.
- **Dynamic Prompt Engineering**: Implements dynamic prompt formatting to enhance the model's understanding and answer generation accuracy.

## 📝 Original Repository

This project is inspired by **ChatPDF** and shares similar functionality. For more details on the original approach, visit:  
👉 [ChatPDF Repository](https://github.com/ali-rose/Bachelor_project)

## 📄 Thesis Summary

Since December 2022, large language models like ChatGPT have revolutionized the AI field. However, these models still suffer from issues like hallucinations and catastrophic forgetting. This project introduces **Retrieval-Augmented Generation (RAG)** to address these challenges by dynamically enhancing the model with relevant external knowledge. With significantly fewer parameters, the model can achieve 63% to 84% of the performance of much larger models, offering a promising approach to solving hallucination issues in specialized fields and promoting privatized AI deployments.

---

> 🧩 This project is intended for research and educational purposes only. All copyrights belong to the original authors and respective institutions.
