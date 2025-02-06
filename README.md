# 🚗👀 Visual Attention Estimation in Drivers 🚗👀
This is the repository for the project of the first module of the course **Electives in AI** held by **Prof. Christian Napoli** at **La Sapienza University of Rome**.
This project is based on this [paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/DGAZE/paper.pdf), which has developed the following dataset and model contained in this [repository](https://github.com/duaisha/DGAZE).  
 

## 🔍 Overview
**Drivers' attention is key to road safety!** 🛑👀 This project aims to estimate where a driver is looking and whether they are paying attention to critical elements on the road. We use **Gaze Point Detection** and **Object Detection** to analyze visual focus and determine attentiveness.

## 🏗️ Project Structure
📂 **`/models`** – Pretrained and fine-tuned models (GazeCNN, Transformer, YOLOv8) 🧠💡  
📂 **`/data`** – DGaze dataset preprocessing and annotations 📊📌  
📂 **`/notebooks`** – Jupyter notebooks for experiments 📖✍️  
📂 **`/scripts`** – Training & evaluation scripts ⚙️🔬  
📂 **`/results`** – Model evaluation & analysis 📈🎯  
📂 **`/docs`** – Project documentation 📜🖊️  

## 🎯 Key Features
✅ **Gaze Estimation** using **CNN + Transformer** & **ResNet-based models** 🏁🚀  
✅ **Object Detection** using **YOLOv8** to identify key road elements 🚘🚦🚶  
✅ **Attention Scoring** to classify attentiveness 🏎️📍  
✅ **DGaze Dataset Analysis** (3761 image pairs) 📊📷  
✅ **Evaluation against state-of-the-art** methods 🏆📌  

## 🏎️ How It Works
1️⃣ **Gaze Point Detection** 🔭👀  
   - Predicts where the driver is looking using deep learning 🧠📍  
2️⃣ **Object Detection** 🚗🛑  
   - YOLOv8 identifies key objects on the road 🏁🚦  
3️⃣ **Attention Analysis** 🎯  
   - Matches the gaze point with detected objects ✅👀  
   - Classifies attentiveness score (0: Not Attentive ❌, 1: Distracted ❗, 2: Focused ✅)  

## 🔥 Model Performance
📌 **Best Model:** **CNN + Transformer (GazeTR-Hybrid)** 🏆🔬  
📌 **Bounding Box Accuracy:** ~46% 🎯📦  
📌 **Mean Absolute Error:** Competitive with state-of-the-art 🤖📈  
📌 **YOLO Performance:** 83.61% Precision, 73.99% Recall 🏎️🚦  

## 🚀 Installation & Usage
```bash
# Clone the repository 🖥️
git clone https://github.com/your-username/VisualAttentionDrivers.git
cd VisualAttentionDrivers

# Install dependencies ⚙️
pip install -r requirements.txt

# Run inference (example script) 🏁
python run_model.py --input_path data/sample_image.jpg
```

## 📌 Results
🧐 **Where do drivers focus the most?**  
- 🚗 **Vehicles (cars & trucks) - 32.1%**  
- 🚶 **Pedestrians - 8.3%**  
- 🏁 **Road Signs - 2.9%**  

## 🎯 Future Work
✅ Improve gaze estimation precision 🎯🔬  
✅ Train on larger datasets for better generalization 📊🌍  
✅ Integrate real-time processing for ADAS 🚗⚡  

## 👥 Contributors
-  [Niccolò Piraino](https://github.com/Nickes10)
-  [Antonio Scardino](https://github.com/antoscardi)
 
