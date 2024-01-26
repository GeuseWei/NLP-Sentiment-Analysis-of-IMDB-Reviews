import os
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import log_loss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# 1. load your training data

# 2. Train your network
# 		Make sure to print your training loss and accuracy within training to show progress
# 		Make sure you print the final training accuracy

# 3. Save your model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_reviews(path, sentiment):
    reviews = []
    labels = []
    # Load all the reviews from the directory.
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
                labels.append(sentiment)
    return reviews, labels


def load_imdb_data(base_path):
    # Define the paths for positive and negative reviews for both training and testing datasets.
    train_pos_path = os.path.join(base_path, 'train', 'pos')
    train_neg_path = os.path.join(base_path, 'train', 'neg')
    test_pos_path = os.path.join(base_path, 'test', 'pos')
    test_neg_path = os.path.join(base_path, 'test', 'neg')

    # Load the reviews and labels for each category.
    train_pos_reviews, train_pos_labels = load_reviews(train_pos_path, 1)
    train_neg_reviews, train_neg_labels = load_reviews(train_neg_path, 0)
    test_pos_reviews, test_pos_labels = load_reviews(test_pos_path, 1)
    test_neg_reviews, test_neg_labels = load_reviews(test_neg_path, 0)

    # Combine the reviews and labels for training and testing datasets.
    train_reviews = train_pos_reviews + train_neg_reviews
    train_labels = train_pos_labels + train_neg_labels
    test_reviews = test_pos_reviews + test_neg_reviews
    test_labels = test_pos_labels + test_neg_labels

    # Create pandas dataframes from the reviews and labels.
    train_data = pd.DataFrame({
        'review': train_reviews,
        'sentiment': train_labels
    })

    test_data = pd.DataFrame({
        'review': test_reviews,
        'sentiment': test_labels
    })
    return train_data, test_data


def preprocess_review(review):
    # Convert to lowercase.
    review = review.lower()
    # Remove non-alphabet characters.
    review = re.sub('[^a-z]', ' ', review)
    # Tokenize the review.
    words = nltk.word_tokenize(review)
    # Remove stop words.
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize the words.
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the preprocessed words back into a string.
    review = ' '.join(words)
    return review


if __name__ == "__main__":

    base_path = "./data/aclImdb"
    train_data, test_data = load_imdb_data(base_path)

    # Preprocess the reviews in the training and testing datasets.
    train_data['review'] = train_data['review'].apply(preprocess_review)
    test_data['review'] = test_data['review'].apply(preprocess_review)

    # Save train_data.
    with open('./data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    # Save test_data.
    with open('./data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)


    # Load train_data.
    with open('./data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    # Load test_data.
    with open('./data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    #
    # Create TF-IDF vectorizer.
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))

    # Fit the vectorizer to the training set reviews and transform them to vectors.
    train_features = vectorizer.fit_transform(train_data['review'])

    # Transform the test set reviews to vectors using the same vectorizer.
    test_features = vectorizer.transform(test_data['review'])

    # Create a Logistic Regression model.
    model = LogisticRegression(solver='liblinear')

    # Train the model with the training set features and labels.
    model.fit(train_features, train_data['sentiment'])

    # Calculate the training accuracy using the model's score() method.
    train_accuracy = model.score(train_features, train_data['sentiment'])

    # Print the training accuracy.
    print('The train accuracy of the Logistic Regression model is:', train_accuracy)

    # Calculate the training loss.
    train_prob = model.predict_proba(train_features)
    train_loss = log_loss(train_data['sentiment'], train_prob)

    # Print the training loss.
    print('The train loss of the Logistic Regression model is:', train_loss)

    # Save the model.
    joblib.dump(model, './models/logistic_regression_model.pkl')

    # Save the vectorizer.
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')

    # Create a SVM model.
    model = svm.SVC(kernel='linear')

    # Train the model with the training set features and labels.
    model.fit(train_features, train_data['sentiment'])

    train_accuracy = model.score(train_features, train_data['sentiment'])

    # Print the training accuracy.
    print('The train accuracy of the SVM model is:', train_accuracy)

    # Save the SVM model.
    joblib.dump(model, '/models/svm_model.pkl')

    # Set the maximum size of the vocabulary.
    max_words = 10000
    # Set the maximum length for each review.
    max_len = 100

    # Create a tokenizer, set the maximum size of the vocabulary.
    tokenizer = Tokenizer(num_words=max_words)
    # Fit the tokenizer using the training set reviews.
    tokenizer.fit_on_texts(train_data['review'])

    # Transform the reviews to sequences of integers using the tokenizer.
    train_sequences = tokenizer.texts_to_sequences(train_data['review'])
    test_sequences = tokenizer.texts_to_sequences(test_data['review'])

    # Pad or truncate the sequences to the same length.
    train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # Define a LSTM model.
    model = Sequential()
    model.add(Embedding(max_words, 50, input_length=max_len))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model.
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    # Train the model.
    history = model.fit(train_sequences, train_data['sentiment'], epochs=10, batch_size=128, validation_split=0.2)

    # Retrieve the accuracy history from the training process.
    accuracy_history = history.history['acc']

    # Get the training accuracy of the last epoch.
    final_training_accuracy = accuracy_history[-1]

    # Print the final training accuracy.
    print('The final training accuracy of the LSTM model is:', final_training_accuracy)

    # Save the trained LSTM model.
    model.save('/models/lstm_model.h5')

    # Save the tokenizer.
    with open('./models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)