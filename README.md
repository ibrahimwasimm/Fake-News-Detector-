🚀 Fake News Detection System

The Fake News Detection System is a machine-learning based web application that predicts whether a given news article is Real or Fake.
The project uses a combination of:

-> Word2Vec embeddings for text vectorization

-> MLP (Multi-Layer Perceptron) classifier for prediction

-> Streamlit for an interactive and visually appealing front-end

-> Cleaned dataset with labels (Fake = 0, Real = 1)

This project is ideal for learning NLP, machine learning pipelines, and deploying ML models as simple web apps.

🧠 Project Features

✔ Clean data preprocessing

✔ Word2Vec model training & vectorization

✔ MLP classifier (trained & saved with pickle)

✔ Real-time text prediction via Streamlit

✔ Gradient-styled UI

✔ Handles long news articles

✔ Displays classification result clearly

✔ Supports copying & pasting large paragraphs


⚙️ Installation & Setup

1. Clone the Repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

3. Create Virtual Environment (Optional)
   
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

5. Install Dependencies
pip install -r requirements.txt

📊 Training the Model

Model training is done in the training.ipynb notebook.

Steps: 
-> Load dataset

-> Clean text (lowercase, remove punctuation, stopwords, etc.)

-> Train Word2Vec model

-> Convert text to vectors

-> Train MLPClassifier

-> Evaluate accuracy, precision, recall

Save models:

pickle.dump(mlp_model, open("mlp_model.pkl","wb"))
word2vec.save("word2vec.model")

🖥️ Running the Streamlit App

From the root directory:
streamlit run app/app.py
