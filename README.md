import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- NLTK Data ---
@st.cache_resource
def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data is ready.")
download_nltk_data()

# --- 1. Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

# --- 2. Data Loading Function (Cached) ---
@st.cache_data
def load_data():
    print("--- Loading Source Data ---")
    try:
        df_existing = pd.read_csv("comments.csv")
        df_toxic_source = pd.read_csv("train.csv")
        return df_existing, df_toxic_source
    except FileNotFoundError:
        st.error("Error: Make sure 'comments.csv' and 'train.csv' are in the same folder.")
        return None, None

# --- 3. Model Training Function (Cached) ---
@st.cache_resource
def train_model():
    print("--- Training Model (v4 - Balanced Dataset) ---")
    df_existing, df_toxic_source = load_data()
    
    if df_existing is None:
        return None, None

    # --- A. Map Original Labels ---
    def map_label_to_category(label):
        if label == 1: return 'praise'
        if label == 2: return 'support'
        if label in [3, 6]: return 'hate_abuse'
        if label in [0, 4, 5]: return 'emotional'
        return None

    df_remapped = pd.DataFrame()
    df_remapped['text'] = df_existing['text']
    df_remapped['category'] = df_existing['label'].apply(map_label_to_category)
    df_remapped = df_remapped.dropna(subset=['category'])

    # --- B. Create High-Quality Manual Data (from all your examples) ---
    print("Creating high-quality augmented data...")
    
    # 1. Praise
    praise_texts = [
        "This is brilliant! You explained a complex topic so simply.",
        "Flawless execution. That was incredibly well done.",
        "Best video I've seen this week. Subscribed!",
        "I learned so much from this. Thank you for making it!",
        "The editing on this is next-level. So clean and professional.",
        "Absolutely legendary. Nothing more to say.",
        "Amazing work! Loved the animation.",
    ] * 50  # Create 350 examples
    df_praise_to_add = pd.DataFrame({'text': praise_texts, 'category': 'praise'})

    # 2. Support
    support_texts = [
        "This deserves way more views.",
        "Keep up the great work, I'm always here to support.",
        "So proud of how much your channel has grown.",
        "Don't worry about the algorithm, just keep making what you love.",
        "It's brave of you to share this, thank you.",
        "Your videos always brighten my day. Please keep going!",
        "Don't let the negative comments get to you. This is valuable content.",
    ] * 50  # Create 350 examples
    df_support_to_add = pd.DataFrame({'text': support_texts, 'category': 'support'})

    # 3. Constructive Criticism
    crit_texts = [
        "The topic is great, but it was hard to hear you over the background music.",
        "You might want to double-check your source for the statistic at 2:30. I think it's outdated.",
        "I love your tutorials, but this one felt a bit rushed. Maybe split it into two parts?",
        "A table of contents with timestamps in the description would be super helpful.",
        "The presentation is good, but the core argument isn't very clear until the end.",
        "I think a different color for the code blocks would make them more readable.",
        "Great video, but the audio quality was a bit echoey. A different mic might help."
    ] * 50  # Create 350 examples
    df_crit_to_add = pd.DataFrame({'text': crit_texts, 'category': 'constructive_criticism'})

    # 4. Hate/Abuse
    hate_texts = [
        "You're a clown. Get a real job.",
        "This is objectively the worst take I have ever heard.",
        "I can't believe anyone actually wastes their time watching this garbage.",
        "Total nonsense from an idiot who thinks they're smart.",
        "Stop making videos. You're just embarrassing yourself.",
        "You have no idea what you're talking about."
    ] * 50  # Create 300 examples
    df_hate_to_add = pd.DataFrame({'text': hate_texts, 'category': 'hate_abuse'})

    # 5. Threat
    # First, get the real examples from train.csv
    df_threat_to_add = pd.DataFrame({'text': df_toxic_source[df_toxic_source['threat'] == 1]['comment_text'], 'category': 'threat'})
    # Now, add your new high-quality examples
    threat_texts = [
        "I'm downloading this before you delete it, so I have proof of how stupid you are.",
        "Keep talking like that and see what happens to your channel.",
        "I know where you live, you should be careful.",
        "I'm reporting this for misinformation.",
        "Say that one more time and I'll make sure your account gets banned.",
        "If you don't take this down, I'm going to flag all your videos."
    ] * 50  # Create 300 examples
    df_threat_aug_to_add = pd.DataFrame({'text': threat_texts, 'category': 'threat'})

    # 6. Emotional
    emo_texts = [
        "I've been struggling with this exact problem. This makes me feel so much less alone.",
        "This is so nostalgic. It brings back so many good memories.",
        "I'm laughing so hard I'm crying. This is hilarious.",
        "This is a really beautiful and touching tribute.",
        "Honestly, this made me feel very anxious, but it's an important topic.",
        "That was incredibly wholesome. I really needed this today.",
        "This actually made me tear up a bit. So powerful."
    ] * 50  # Create 350 examples
    df_emo_to_add = pd.DataFrame({'text': emo_texts, 'category': 'emotional'})

    # 7. Irrelevant/Spam
    spam_texts = [
        "add me on snap [username]",
        "Who's watching this in 2026?",
        "Awesome video! Anyway, I just released a new song, check out my channel!",
        "Can you make a video about [something completely unrelated]?",
        "Click here to earn $1000 a day -> [suspicious link]",
        "Like this comment if you agree.",
        "Check out my profile for amazing deals!",
        "First!",
        "www. buy-this-scam. com",
        "sub for sub? I subscribed to you."
    ] * 50  # Create 500 examples
    df_spam_to_add = pd.DataFrame({'text': spam_texts, 'category': 'spam_irrelevant'})
    
    # 8. Question/Suggestion
    question_texts = [
        "What software do you use to make this?",
        "Can you make a video on topic X next?",
    ] * 50  # Create 100 examples
    df_question_to_add = pd.DataFrame({'text': question_texts, 'category': 'question_suggestion'})

    # --- C. Undersample the Original Data ---
    print("Undersampling original dataset...")
    # We'll take 2000 random samples from each of the big categories
    sample_size = 2000 
    df_praise_orig = df_remapped[df_remapped['category'] == 'praise'].sample(sample_size)
    df_support_orig = df_remapped[df_remapped['category'] == 'support'].sample(sample_size)
    df_hate_orig = df_remapped[df_remapped['category'] == 'hate_abuse'].sample(sample_size)
    df_emo_orig = df_remapped[df_remapped['category'] == 'emotional'].sample(sample_size)


    # --- D. Combine Everything into One BALANCED DataFrame ---
    print("Combining all datasets...")
    df = pd.concat([
        # Original (but sampled) data
        df_praise_orig,
        df_support_orig,
        df_hate_orig,
        df_emo_orig,
        
        # Your new high-quality data
        df_threat_to_add,
        df_praise_to_add,
        df_support_to_add,
        df_crit_to_add,
        df_hate_to_add,
        df_threat_aug_to_add,
        df_emo_to_add,
        df_spam_to_add,
        df_question_to_add
    ], ignore_index=True)
    
    print(f"Total rows in new balanced dataset: {len(df)}")
    print("\nNew balanced category distribution:")
    print(df['category'].value_counts())

    # --- Preprocess Text ---
    print("\nPreprocessing balanced dataset...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # --- Train Model ---
    X = df['cleaned_text']
    y = df['category']
    
    vectorizer = TfidfVectorizer(max_features=5000) # Use 5000 features
    X_train_tfidf = vectorizer.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y)
    
    print("--- Model Training Complete ---")
    return model, vectorizer

# --- 4. Reply Suggestion Function ---
def get_reply_suggestion(category):
    templates = {
        'praise': "Thank you so much for the kind words! We're thrilled you enjoyed it.",
        'support': "Thank you for the support, it means a lot to us!",
        'hate_abuse': "[Action: Monitor user or escalate to moderation.]",
        'emotional': "Thank you for sharing that with us. It means a lot.",
        'threat': "[Action: Escalate to security/legal team immediately.]",
        'constructive_criticism': "That's valuable feedback. We'll pass it to the team.",
        'spam_irrelevant': "[Action: Remove comment and monitor user.]",
        'question_suggestion': "That's a great question! We'll look into it."
    }
    return templates.get(category, "No suggestion available.")

# --- 5. Build the Streamlit App UI ---
st.title("Comment Categorization & Reply Assistant 💬")
st.markdown("Enter a comment below to classify its category and get a suggested reply.")

# Load the model and vectorizer
model, vectorizer = train_model()

# Create the text area for user input
user_comment = st.text_area("Enter comment here:")

# Create a button to classify
if st.button("Classify Comment"):
    if user_comment:
        if model and vectorizer:
            cleaned_comment = preprocess_text(user_comment)
            comment_tfidf = vectorizer.transform([cleaned_comment])
            prediction = model.predict(comment_tfidf)[0]
            reply = get_reply_suggestion(prediction)
            
            st.subheader(f"Predicted Category: {prediction.title()}")
            st.info(f"**Suggested Reply:**\n{reply}")
        else:
            st.error("Model could not be loaded. Please check file paths.")
    else:
        st.warning("Please enter a comment to classify.")
