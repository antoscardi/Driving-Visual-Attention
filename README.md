# Visual Attention Estimation in Drivers
This is the repository for the project of the first module of the course **Electives in AI** held by **Prof. Christian Napoli** at **La Sapienza University of Rome**.
This project is based on this [paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/DGAZE/paper.pdf), whose authors also provide this [repository](https://github.com/duaisha/DGAZE). 

## 🔍 Overview
**Drivers' attention is key to road safety!** 🚗. This project aims to estimate where a driver is looking at 👀 and whether they are paying attention to critical elements on the road. We use **Gaze Point Detection** and **Object Detection** to analyze visual focus and determine attentiveness.

We first develop a baseline model, **a CNN-based architecture inspired by GazeCNN**, which takes as input facial features such as eye position, nose position, and head pose to estimate the gaze point.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/23eb83dd-4938-4d52-8398-620a3be0fa73" width="60%">
</p>

And a **larger model** which includes the use of a **Transformer**, specifically a ResNet + Transformer hybrid (GazeTR-Hybrid), which leverages self-attention mechanisms to improve gaze estimation accuracy.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b593079-5aff-4a4a-a550-f7b62451dab0" width="60%">
</p>

At the end, what we obtain is an **attention score that determines the level of attentiveness, a bounding box for the identified object of focus, and an assessment of whether the driver is paying attention to critical road elements.**  

<p align="center">
  <img src="https://github.com/user-attachments/assets/766fdea6-2e3a-432d-b053-cf5f37e3acec" width="50%">
</p>

## 🎯 Key Features
✅ **Gaze Estimation** using **CNN + Transformer** & **ResNet-based models** 🏁🚀  
✅ **Object Detection** using **YOLOv8** to identify key road elements 🚘🚦🚶  
✅ **Attention Scoring** to classify attentiveness 🏎️📍  
✅ **DGaze Dataset Analysis** (3761 image pairs) 📊📷  
✅ **Evaluation against state-of-the-art** methods 🏆📌  

## 🏎️ How It Works
- 🔭👀 **Gaze Point Detection**: Predicts where the driver is looking using deep learning models 🧠  
- 🚗🛑 **Object Detection**: YOLOv8 identifies key objects on the road 🚦  
- 🎯 **Attention Analysis**: Classifies attentiveness score (0: Not Attentive ❌, 1: Distracted ❗, 2: Focused ✅)  

## 🔥 Model Performance
📌 **Best Model:** **CNN + Transformer (GazeTR-Hybrid)** 🏆  
📌 **Bounding Box Accuracy:** ~46% 🎯📦  
📌 **Mean Absolute Error:** Competitive with state-of-the-art 📈  
📌 **YOLO Performance:** 83.61% Precision, 73.99% Recall 🏎️  

## 🚀 Installation & Usage
The dataset was provided by the authors of the paper and without it the models cannot be trained.
```bash
# Clone the repository 🖥️
git clone https://github.com/your-username/VisualAttentionDrivers.git
cd 

# Install dependencies ⚙️
pip install -r requirements.txt

# Run inference (example script) 🏁
python run_model.py --input_path data/sample_image.jpg
```

## 🎯 Results
🧐 **Where do drivers focus the most?**  
- 🚗 **Vehicles (cars & trucks) - 32.1%**  
- 🚶 **Pedestrians - 8.3%**  
- 🛑 **Road Signs - 2.9%**  

## 👥 Contributors
-  [Niccolò Piraino](https://github.com/Nickes10)
-  [Antonio Scardino](https://github.com/antoscardi)
 
