# Offensive Persian Word Detection (ParsBERT + BiLSTM)

This project implements a **hate speech / offensive language detection model for Persian (Farsi)** text.  
It combines **ParsBERT** (pretrained transformer from [HooshvareLab](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)) with a **BiLSTM layer** and modern deep learning techniques to achieve strong performance on Persian offensive language detection.

---

##  Features
-  **ParsBERT backbone** for high-quality Persian embeddings.  
-  **Data augmentation**: synonym replacement & word dropout.  
-  **Class balancing**: oversampling with `imblearn`.  
-  **BiLSTM + pooling layers**: improves contextual understanding.  
-  **Mixed precision training**: faster training with lower memory usage.  
-  **Learning rate decay + AdamW optimizer** for better convergence.  
-  **Model checkpointing + early stopping** for stability.  

---

## Dataset
The project expects a CSV file (example: `Dataset_with_span_and_target.csv`) with the following columns:

- `text` → the Persian text sample  
- `HateSpeech` → binary label (`0 = normal`, `1 = offensive`)  
