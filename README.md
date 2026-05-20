# SMS Spam Detection Using NLP with Neural Network

## Overview
This project is an AI-powered SMS Spam Detection system that classifies text messages as Spam or Ham (Legitimate) using Natural Language Processing (NLP) and a Neural Network model.

The system preprocesses SMS text, converts it into numerical features, and predicts whether the message is spam.

---

## Features
- SMS spam classification
- NLP preprocessing pipeline
- TF-IDF vectorization
- Neural Network model using TensorFlow/Keras
- Real-time prediction support
- Simple and scalable architecture
- Flask/FastAPI web application support
- Streamlit interactive interface
- Deployment support for Render and Hugging Face Spaces
- Advanced deep learning models including LSTM and Transformers
- Email spam detection capability
- Real-time API-based spam prediction
- User authentication and dashboard support
- MongoDB/PostgreSQL scalable database integration
- Docker containerization support
- CI/CD deployment pipeline compatibility
- Multilingual spam and phishing detection
- Model explainability and visualization tools
- Production monitoring and logging support

---

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Flask (optional for deployment)

---

## Dataset
The project uses labeled SMS datasets containing spam and ham messages.

Example:

| Message | Label |
|---------|------|
| Congratulations! You won a prize | Spam |
| Are we meeting today? | Ham |

---

## Model Workflow
1. Data Cleaning
2. Tokenization
3. Stopword Removal
4. TF-IDF Feature Extraction
5. Neural Network Training
6. Model Evaluation
7. Real-Time Prediction

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Results
The model achieved strong classification performance with high spam detection accuracy and low false-negative rates.

---

## Future Improvements
- Add multilingual spam detection
- Deploy using cloud services
- Build mobile integration
- Improve phishing detection capability

---

## Author
Navya S  
B.Sc Computer Science with Artificial Intelligence

---

## Project Structure

```text
sms-spam-detection-nlp/
|-- data/
|   `-- spam.csv
|-- notebooks/
|   `-- model_training.ipynb
|-- src/
|   |-- preprocess.py
|   |-- train_model.py
|   |-- predict.py
|   `-- app.py
|-- models/
|   `-- spam_classifier.h5 (generated after training)
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/train_model.py
```

4. Run API locally:

```bash
python src/app.py
```

---

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/sms-spam-detection-nlp.git
git push -u origin main
```

Replace `yourusername` with your GitHub username.
