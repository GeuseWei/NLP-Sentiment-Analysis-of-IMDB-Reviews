# import required packages
import joblib
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# 1. Load your saved model

# 2. Load your testing data

# 3. Run prediction on the test data and print the test accuracy

if __name__ == "__main__":

    # Load test_data
    with open('./data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print('Start evaluating Logistic Regression model...')

    # Load the vectorizer.
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

    # Transform the test set reviews into vectors using the same vectorizer.
    test_features = vectorizer.transform(test_data['review'])

    # Load the Logistic Regression model.
    logistic_model = joblib.load('./models/logistic_regression_model.pkl')

    # Use the trained model to predict the sentiment of the test set reviews.
    logistic_predictions = logistic_model.predict(test_features)

    # Calculate the prediction accuracy.
    logistic_accuracy = accuracy_score(test_data['sentiment'], logistic_predictions)

    print('The test accuracy of the Logistic Regression model is:', logistic_accuracy)


    print('Start evaluating SVM model...')

    # Load the SVM model.
    svm_model = joblib.load('./models/svm_model.pkl')

    # Predict the sentiment of the test set reviews using the SVM model.
    svm_predictions = svm_model.predict(test_features)

    # Calculate the prediction accuracy.
    svm_accuracy = accuracy_score(test_data['sentiment'], svm_predictions)

    print('The test accuracy of the SVM model is:', svm_accuracy)


    print('Start evaluating LSTM model...')
    max_len = 100

    # Load the LSTM model.
    lstm_model = load_model('./models/lstm_model.h5')

    # Load the tokenizer.
    with open('./models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Now you can use the loaded model and tokenizer to predict the sentiment of new reviews.
    test_sequences = tokenizer.texts_to_sequences(test_data['review'])
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    test_loss, test_acc = lstm_model.evaluate(test_sequences, test_data['sentiment'])
    print('The test accuracy of the LSTM model is:', test_acc)