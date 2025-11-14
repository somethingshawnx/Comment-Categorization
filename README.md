# Comment Categorization & Reply Assistant 💬

An intelligent comment classification system that automatically categorizes user comments into 8 distinct categories and provides suggested replies for content moderation and engagement.

## 🎯 Features

- **Multi-Category Classification**: Classifies comments into 8 categories:
  - Praise
  - Support
  - Constructive Criticism
  - Hate/Abuse
  - Threat
  - Emotional
  - Spam/Irrelevant
  - Question/Suggestion

- **Smart Reply Suggestions**: Provides context-appropriate reply templates for each category
- **Real-time Classification**: Instant classification through an interactive web interface
- **Balanced Training Dataset**: Uses a carefully curated dataset with 10,000+ samples across all categories

## 📋 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - streamlit
  - pandas
  - scikit-learn
  - nltk

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data files**
   
   Place the following CSV files in the project root directory:
   - `comments.csv` - Your existing labeled comment dataset
   - `train.csv` - Toxic comments dataset (for threat category)

   **Required columns:**
   - `comments.csv`: `text`, `label` (0-6)
   - `train.csv`: `comment_text`, `threat`

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. Enter a comment in the text area
2. Click "Classify Comment"
3. View the predicted category and suggested reply

### Example Classifications

```python
# Praise
"This is brilliant! You explained a complex topic so simply."
→ Category: praise
→ Reply: "Thank you so much for the kind words! We're thrilled you enjoyed it."

# Threat
"I'll report you if this continues."
→ Category: threat
→ Reply: "[Action: Escalate to security/legal team immediately.]"

# Constructive Criticism
"Great video, but the audio quality was a bit echoey."
→ Category: constructive_criticism
→ Reply: "That's valuable feedback. We'll pass it to the team."
```

## 🔧 Development

### Training the Model

The model is trained automatically when you run the Streamlit app for the first time. The training process:

1. Loads and remaps existing labeled data
2. Adds high-quality augmented examples for each category
3. Balances the dataset through undersampling
4. Preprocesses text (lowercasing, lemmatization, stopword removal)
5. Trains a Logistic Regression model with TF-IDF features

### Jupyter Notebook

Use `project.ipynb` for:
- Model experimentation
- Performance evaluation
- Dataset analysis
- Custom testing

Run the notebook cells in order to:
1. Load and prepare data
2. Preprocess text
3. Train the model
4. Evaluate performance
5. Test individual comments

## 📊 Model Performance

The model achieves approximately:
- **Overall Accuracy**: ~92%
- **F1-Score**: 0.89 (macro average)

Category-specific performance:
- Praise: 0.93 F1-score
- Support: 0.79 F1-score
- Emotional: 0.95 F1-score
- Hate/Abuse: 0.90 F1-score
- Constructive Criticism: 1.00 F1-score
- Spam/Irrelevant: 1.00 F1-score
- Threat: 0.63 F1-score

## 🏗️ Project Structure

```
.
├── app.py                 # Streamlit web application
├── project.ipynb          # Jupyter notebook for training & testing
├── requirements.txt       # Python dependencies
├── comments.csv          # Your labeled comment dataset
└── train.csv             # Toxic comments dataset
```

## 🔍 How It Works

### Text Preprocessing
1. Convert to lowercase
2. Remove punctuation and numbers
3. Tokenization
4. Remove stopwords
5. Lemmatization

### Feature Extraction
- TF-IDF Vectorization with 5,000 features
- Captures word importance across documents

### Classification
- Logistic Regression classifier
- Multi-class classification (one-vs-rest)
- Trained on balanced dataset

## 🤝 Contributing

Contributions are welcome! To improve the model:

1. Add more training examples to the augmented data sections
2. Experiment with different preprocessing techniques
3. Try alternative classification algorithms
4. Enhance reply templates

## 📝 Label Mapping

Original labels (0-6) are mapped as follows:
- `0` (sadness) → emotional
- `1` (joy) → praise
- `2` (love) → support
- `3` (anger) → hate_abuse
- `4` (fear) → emotional
- `5` (surprise) → emotional
- `6` (toxic) → hate_abuse

## ⚠️ Important Notes

- The model requires both `comments.csv` and `train.csv` to function
- First run downloads NLTK data (stopwords, wordnet)
- Training happens automatically and is cached for performance
- Threat category has lower recall due to limited training data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 👥 Authors

[Add your name/team here]

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- scikit-learn for machine learning algorithms
- Streamlit for the web interface framework
