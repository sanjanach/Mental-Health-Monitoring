import praw
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from dotenv import load_dotenv
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, GlobalMaxPooling1D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import gensim.downloader as api
from sklearn.utils import class_weight

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def setup_reddit_api():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

def fetch_reddit_data(reddit, subreddits, limit=8800):
    posts = []
    for subreddit_name in subreddits:
        print(f"Fetching from r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.top('year', limit=limit // len(subreddits)):
            posts.append({ 'title': submission.title, 'text': submission.selftext, 'subreddit': subreddit_name })
    return pd.DataFrame(posts)

mental_health_categories = {
    'SuicideWatch': 'mental_health',
    'schizophrenia': 'mental_health',
    'depression': 'mental_health',
    'anxiety': 'mental_health',
    'BPD': 'mental_health',
    'bipolar': 'mental_health',
    'ptsd': 'mental_health',
    'mentalhealth': 'mental_health',

    'happy': 'stable',
    'CasualConversation': 'stable',
    'advice': 'stable'
}

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.mental_health_stopwords = {'feel', 'feeling', 'think', 'really', 'know', 'like', 'get', 'want', 'would', 'could', 'should'}
        self.stop_words.update(self.mental_health_stopwords)
    
    def preprocess(self, text):
        if pd.isna(text) or text == "": 
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def process_dataframe(self, df):
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['full_text'].apply(self.preprocess)
        
        df = df[df['processed_text'].str.len() > 10].reset_index(drop=True)
        df = df[df['processed_text'] != ''].reset_index(drop=True)
        
        df['mental_health_category'] = df['subreddit'].map(lambda x: mental_health_categories.get(x, 'unknown'))
        return df

def train_deep_learning_model(df):
    MAX_NB_WORDS = 15000  
    MAX_SEQUENCE_LENGTH = 200
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_text'].values)
    word_index = tokenizer.word_index
    
    X = tokenizer.texts_to_sequences(df['processed_text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['mental_health_category'])
    Y_encoded = tf.keras.utils.to_categorical(integer_encoded)
    class_names = list(label_encoder.classes_)
    
    X_train, X_test, Y_train_encoded, Y_test_encoded = train_test_split(
        X, Y_encoded, test_size=0.2, random_state=42, stratify=integer_encoded)
    
    y_train_labels = np.argmax(Y_train_encoded, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {class_weights_dict}")

    print("Loading Word2Vec model...")
    word2vec_model = api.load('word2vec-google-news-300')
    EMBEDDING_DIM = word2vec_model.vector_size
    
    embedding_matrix = np.random.normal(0, 0.1, (len(word_index) + 1, EMBEDDING_DIM))
    found_words = 0
    for word, i in word_index.items():
        if word in word2vec_model:
            embedding_matrix[i] = word2vec_model[word]
            found_words += 1
    print(f"Found {found_words}/{len(word_index)} words in Word2Vec model")
    
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    
    embedding = Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                         weights=[embedding_matrix], 
                         input_length=MAX_SEQUENCE_LENGTH, 
                         trainable=True)(input_layer)
    
    embedding = SpatialDropout1D(0.2)(embedding) 
    
    lstm1 = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embedding)
    lstm2 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(lstm1)
    
    pooled = GlobalMaxPooling1D()(lstm2)
    
    dense1 = Dense(128, activation='relu')(pooled)
    dense1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    
    output = Dense(Y_encoded.shape[1], activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        X_train, Y_train_encoded, 
        epochs=15,
        batch_size=32, 
        validation_split=0.15,  
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    model = tf.keras.models.load_model('models/best_model.keras')
    
    accr = model.evaluate(X_test, Y_test_encoded, verbose=0)
    print(f'Test set\n  Loss: {accr[0]:.4f}\n  Accuracy: {accr[1]:.4f}')
    
    Y_pred_probs = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_probs, axis=1)
    Y_test_labels = np.argmax(Y_test_encoded, axis=1)
    
    print('\nClassification Report:')
    print(classification_report(Y_test_labels, Y_pred, target_names=class_names))
    
    model.save('models/mental_health_bilstm_model.keras')
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    with open('models/label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle)
    
    return accr[1]

def run_main_pipeline():
    print("Starting Enhanced Mental Health Monitoring Pipeline...")
    subreddits = ['depression', 'anxiety', 'SuicideWatch', 'BPD', 'bipolar', 'ptsd', 'schizophrenia', 'mentalhealth', 'happy', 'CasualConversation', 'advice']
    
    data_path = "data/reddit_mental_health_data_large.csv"
    
    if not os.path.exists(data_path):
        print("No local data found. Fetching from Reddit API...")
        try:
            reddit_api = setup_reddit_api()
            reddit_df = fetch_reddit_data(reddit_api, subreddits)
            reddit_df.to_csv(data_path, index=False)
        except Exception as e:
            print(f"Error collecting data: {e}. Exiting.")
            return
    else:
        reddit_df = pd.read_csv(data_path)
    
    print(f"\nLoaded {len(reddit_df)} Reddit posts.")
    
    print("\nPreprocessing text data...")
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(reddit_df)
    
    print(f"After preprocessing: {len(processed_df)} posts")
    print("\nClass distribution:")
    print(processed_df['mental_health_category'].value_counts())
    
    print("\nTraining Enhanced Deep Learning model...")
    accuracy = train_deep_learning_model(processed_df)
    
    print("\nPipeline completed successfully!")
    print(f"Final Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_main_pipeline()
