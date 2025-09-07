# 🚦 Traffic Sign Classifier

This project is a deep learning application that classifies German traffic signs using **TensorFlow / Keras** and **MobileNetV2**.  
Uploading "Screenshot 2025-09-07 214210.png"... 
## 📌 Project Overview
- Dataset: [GTSRB - German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)  
- Goal: Train a CNN model to recognize and classify different traffic signs.  
- Frameworks: TensorFlow, Keras, Streamlit.  

## ⚙️ Steps Implemented
1. **Data Preparation**  
   - Selected a subset of traffic sign classes.  
   - Organized dataset into `train/val/test` using `splitfolders`.  

2. **Model Training**  
   - Used **MobileNetV2** (pretrained on ImageNet) as a feature extractor.  
   - Added dense layers for classification.  
   - Trained with categorical cross-entropy loss.  
   - Achieved high accuracy on test data.  

3. **Saving the Model**  
   - Final trained model saved as `traffic_model.h5`.  

4. **Streamlit App**  
   - Interactive web app where users can upload a traffic sign image.  
   - The app displays:  
     - Predicted traffic sign label.  
     - Confidence score.  
```bash
