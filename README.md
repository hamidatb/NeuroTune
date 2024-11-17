# 🎵 NeuroTune: Redefining Music Accessibility for all Abilities 🎵

Music is one of the most universal sources of comfort and joy. For many of us, the process of finding the right playlist is simple: we open our music app and search for "happy playlist" or "sad playlist." But have you ever considered how many steps this actually involves? Or the level of fine motor control required to navigate through these apps?

Now imagine someone with a disability that might impact motor control and/or speech, such as cerebral palsy or Parkinson's. For people with these conditions, the simple task of picking and playing a song becomes overwhelming. Technology has the power —and responsibility— to bridge this gap. **NeuroTune** was built with one mission: to ensure everyone can experience the same joy and comfort of music, regardless of physical or mental limitations.

## 🌟 **What Is NeuroTune?**
**NeuroTune** is a groundbreaking application designed to empower individuals with a large range of motor and speech disabilities:
- **Real-Time Eye-Tracking** to enable effortless navigation.
- **Mood Detection with EEG Data** from the Muse S device to create personalized music experiences.

This hands-free system provides users with instant access to tailored playlists based on their emotions, making music more accessible and impactful. Users can navigate through the application by simply focusing their gaze on an option for 3 seconds. The **🟢 green circle cursor** provides intuitive, real-time feedback as it tracks their eye movements, ensuring a smooth and interactive experience.

<img src="static/media/screenshots/NeuroTune Thumbnail.png" alt="Neurotune Thumbnail" width="500">  

## 💡 **Key Features**

### 🎯 **Real-Time Eye Tracking**
We trained our **custom eye-tracking neural network** to map gaze positions to specific screen quadrants. Users can navigate the app with just their eyes, no need for physical touch or verbal instructions. All they need is a focused gaze.

### 🧠 **Mood Detection**
Using EEG data collected via the **Muse S device**, we trained a neural network to classify emotional states like happiness, sadness, relaxation, and stress. NeuroTune uses these insights to tailor playlists that match the user’s mood.

### 🎵 **Playlist Integration**
The app connects directly with a custom playlist to fetch and play personalized songs based on the detected mood. Whether you're feeling joyful or reflective, NeuroTune brings you the perfect soundtrack.

### 💻 **Accessible and Aesthetic UI**
- **Large readable fonts** and **constrastive colors** ensure the interface is intuitive, accessible, and welcoming.
- A distinct **green circle cursor** smoothly transitions across the screen, showing users exactly where their gaze is focused.
- Real-time visual feedback highlights active quadrants to enhance accessibility.

### 🌐 **Hands-Free Navigation**
Users can control the app entirely with their eyes:
- Move your gaze to navigate between interface options.
- Engage in mood detection and enjoy therapeutic music without lifting a finger.


## ⚙️ **Technology Stack**

### **Core Machine Learning**
- **Custom Eye-Tracking Model**: Trained on relative pupil positions to accurately predict screen quadrants.
- **Muse S Emotion Model**: Maps EEG signals from the Muse S headband to four core emotional states- happiness, sadness, relaxation, and stress.

### **Frontend**
- **HTML5**, **CSS3**, **JavaScript**
- **Socket.IO** for real-time bidirectional communication between the server and the browser.
- Smooth animations for gaze-based interaction.

### **Backend**
- **Flask**: Python-based web framework for serving the application.
- **Flask-SocketIO**: Enables real-time updates to the UI based on gaze predictions.

### **Computer Vision**
- **OpenCV**: Captures and processes real-time video streams for eye-tracking.
- **Dlib**: Detects facial landmarks to isolate the eyes and calculate pupil positions.

### **Tools & Libraries**
- **Pykan**: Import the necessary libraries for the KAN model.
- **NumPy**: Numerical computations for machine learning models.
- **Scikit-learn**: Training and evaluating neural networks.
- **Pickle**: For model serialization.
- **Eventlet**: Asynchronous capabilities for real-time Flask apps.

## 📸 **Screenshots of the Application**

### **Homepage**
<img src="static/media/screenshots/homepage.png" alt="Homepage" width="500">  
<p> 🟢 green circle cursor: The eye-tracking cursor moves with your gaze, providing real-time visual feedback. </p>

### **Mood Detection Screen**
<img src="static/media/screenshots/mood_detection0.png" alt="Mood Detection 0" width="500">  

<img src="static/media/screenshots/mood_detetction1.png" alt="Mood Detection 1" width="500">

### **Music Player of the Happy Playlist**
<img src="static/media/screenshots/happy_playlist.png" alt="Happy Playlist Screenshot" width="500">  

### **A User Using NeuroTune**
<img src="static/media/screenshots/user_using_Neurotune.png" alt="User Using Neurotune" width="500">  


## 📊 **Machine Learning Visualizations**

Explore the insights behind NeuroTune's advanced models:

### **Mood Detection Model (KAN)**
- Our custom **KAN (Kolmogorov-Arnold Network)** was trained using EEG data from the Muse S device, enabling precise mood classification.
- Below are some visualizations showcasing the model architecture, test results and accuracy, and extracted statistics:

###### **KAN Architecture**
<img src="static/ml_visualizations/KAN_better.png" alt="KAN Architecture" width="500"> 

###### **Test Results and Accuracy**
<img src="static/ml_visualizations/KAN_test.png" alt="Test Results and Accuracy" width="500"> 

###### **Relative BrainWave Types for Moods**
<img src="static/ml_visualizations/RelativeBrainwaveTypeForMoods.png" alt="Relative BrainWave Types for Moods" width="500">  

###### **Multivariable Correlation Between Brain Types and Moods**
<img src="static/ml_visualizations/MVCorrBetweenBrainwaveTypeAndMood.png" alt="Multivariable Correlation Between Brain Types and Moods" width="500">  

- Custom-trained on relative pupil positions to predict screen quadrants with high accuracy.
- Below are visualizations demonstrating its performance:

###### **Quadrant Prediction Training**
<img src="static/ml_visualizations/training_gaze_model.png" alt="Quadrant Prediction Training" width="500">  
---


## 🛠 **How We Built It**

1. **Training the Mood Detection Model**
   - We collected EEG data using the Muse S headband across different emotional states.
   - Using this labelled data, we trained a neural network to classify emotions like happiness, sadness, stress, and relaxation. 
   - Here, we employed Jupyter Notebooks and Google Colab to explore many models, before settling on a KAN   

2. **Training the Eye-Tracking Model**
   - Captured over 200 images of gaze positions for each screen quadrant.
   - Extracted relative pupil positions and trained a custom eye-tracking model to map these positions to screen quadrants.

3. **Real-Time Communication**
   - Integrated Flask-SocketIO to send gaze predictions from the backend to the browser in real-time.
   - The frontend responds instantly to user gaze with cursor movement and quadrant highlights.

4. **Seamless User Experience**
   - Designed a simple, visually pleasing UI with large fonts and intuitive navigation.
   - The cursor movement and quadrant highlights provide immediate feedback, making the app feel responsive and alive.


## 🚀 **How to Get Started**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuroTune.git
   cd NeuroTune
   ```

2. Set up the environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Calibrate the gaze model on yourself:
   ```bash
   python utils/train_quadrant_model.py
   ```

4. Start the server:
   ```bash
   python app.py
   ```

5. Access the app at `http://localhost:5000` and enjoy!

## 🚀 Contributors

- **Hamidat Bello**: Project Lead. Developed the backend and frontend, custom-built and trained the SVM eye-tracking model using Scikit-learn, and contributed to the UX design.
- **Chidinma Obi-Okoye**: Contributed to the software, working on both the backend and frontend, debugging, and creating the Figma wireframes for the UX design.
- **Ezra Dese**: Helped develop the KAN EEG mood model, worked on the Muse S EEG hardware, and integrated the mood detection model into the Flask app. Led model refinement and filtering.
- **Emmanuel Ssempala**: Focused on developing the KAN EEG mood model and assisted with the Muse S EEG hardware integration.
- **Onami Ademola**: Worked on the Muse S EEG hardware and contributed to the development of the KAN EEG mood model. Created the script to show the MUSE S Channel reading when in debug mode.
- **Katie Bauer**: Led the development, refinement, and training of the KAN EEG mood model. Researched other possible model types to use.


## 🤝 **Contributing**
We welcome contributions from the community! Whether it’s improving the eye-tracking model, optimizing the UI, or suggesting new features, your input is valued.

---

### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.  

---

Let’s make music accessible for all. 🎶 Together, we can change lives.


