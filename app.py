import numpy as np
from flask import Flask, request,render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer


import heapq
import re
import string
import random
import pickle



import joblib
from keras import models


SEQUENCE_LENGTH = 8

# Create flask app
flask_app = Flask(__name__)
# model = pickle.load(open("lstm_model.pickle", "rb"))
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vectorizer = joblib.load('vectorizer.pkl')  # load SENTIMENT vectorizer
lemmatizer = WordNetLemmatizer()



with open('char_indices_last.pickle', 'rb') as f:
    # Load the dictionary from the file using pickle.load()
    char_indices = pickle.load(f)

# print(char_indices)


with open('indices_char_last.pickle', 'rb') as f:
    # Load the dictionary from the file using pickle.load()
    indices_char = pickle.load(f)

# print(indices_char)

model = models.load_model('256-128-128(relu)-40(D)-8-last-1.h5')

with open("cfdist.pkl", "rb") as f:
    cfdist = pickle.load(f)
model_ngram = cfdist

# lstm_model = pickle.load(open('lstm__model.pkl', 'rb'))
# with open('lstm_model.pkl', 'rb') as f:
#     lstm_model = pickle.load(f)

@flask_app.route("/")
def Home():
    return render_template("index.html")



@flask_app.route("/predict", methods=["POST"])
def predict():
    SEQUENCE_LENGTH = 8
    model = models.load_model('256-128-128(relu)-40(D)-8-last-1.h5')
    model_LR = joblib.load('modellr.pkl')  # load sentiment LR model
    # n_words = 1
    # max_length = 3

    seed_text = request.form['words']

    # Check which button was clicked based on its name
    if 'predict_word' in request.form:
        # Perform prediction for next word
        generated_text = predict_completions(seed_text,model,SEQUENCE_LENGTH,n=5 )
        return render_template('index.html', prediction_text=generated_text,input_text=seed_text)

    elif 'sentiment_analysis' in request.form:
        # Perform sentiment analysis
        sentiment_text, probability_dict, label_with_max_value = sentiments(
            model_LR, seed_text)
        return render_template('index.html', sentiment_of_text=sentiment_text, sent_prob=probability_dict, input_text=seed_text, result_sentiment=label_with_max_value)
    elif 'tri-gram' in request.form:
        n_gram_text,possible_list= ngram_predict(seed_text,model_ngram)
        return render_template('index.html', prediction_text_ng=n_gram_text,possible_list=possible_list ,input_text=seed_text)
        



def preprocess(text):
    lowerstr = text.lower()
    # Tokenize the text
    words = word_tokenize(lowerstr)
    # Remove stopwords and lemmatize the remaining words
    words = [lemmatizer.lemmatize(
        word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)


def sentiments(model_LR, seed_text):
    seed_text = preprocess(seed_text)
    predictions = model_LR.predict(vectorizer.transform([seed_text]))
    probabilities = model_LR.predict_proba(vectorizer.transform([seed_text]))
    # for i, label in enumerate(model_LR.classes_):
    #       print(f"Probability of {label}: {probabilities[0][i]}")
    result_dict = {}
    for i, label in enumerate(model_LR.classes_):
        result_dict[label] = probabilities[0][i]
    label_with_max_value = max(result_dict, key=result_dict.get)
    return predictions, result_dict,label_with_max_value


def predict_completions(text, model, SEQUENCE_LENGTH, n= 3):
        text = text[-SEQUENCE_LENGTH:].lower()
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, n)
        return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

#Prediction function


def predict_completion(text):
    original_text = text
    generalised = text
    completion = ''
    i=0
    max_iterations = 20
    while i<max_iterations:
        
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]

        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 4 > len(original_text) and next_char == ' ':
            print(completion)
            return completion
        i=i+1
        if i==max_iterations-1:
            return completion


#The sample function
#This function allows us to ask our model what are the next probable characters (The heap simplifies the job)
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

#Testing


def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, 40))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1
    return x


# def ngram_predict(user_input, model_ngram):
#     user_input = filter1(user_input)
#     user_input = user_input.split()
#     w1 = len(user_input) - 2
#     w2 = len(user_input)
#     prev_words = user_input[w1:w2]

#     # display prediction from highest to lowest maximum likelihood
#     try:
#         prediction = sorted(dict(model_ngram[prev_words[0], prev_words[1]]), key=lambda x: dict(
#             model_ngram[prev_words[0], prev_words[1]])[x], reverse=True)
#         print("Trigram model predictions: ", prediction)
#         word = []
#         weight = []
#         c = dict(model_ngram[prev_words[0], prev_words[1]])
#         for key, prob in dict(model_ngram[prev_words[0], prev_words[1]]).items():
#             print(key, prob)
#             word.append(key)
#             weight.append(prob)
#         keymax = max(zip(c.values(), c.keys()))[1]
#         print(keymax)
#         spaces=[]
#         for i in prediction:
#             spaces.append(" "+i)
#         prediction=spaces
#         # pick from a weighted random probability of predictions
#         next_word = random.choices(word, weights=weight, k=1)
#         # add predicted word to user input
#     #     user_input.append(next_word[0])
        
#         # user_input.append(keymax)
        
#         return ( keymax,prediction[:5])
#     except IndexError :
#         return('<unk>',['<unk>'])

    


# def filter1(text):
#     # normalize text
#     # text = (unicodedata.normalize('NFKD', text).encode(
#     #     'ascii', 'ignore').decode('utf-8', 'ignore'))
#     # replace html chars with ' '
#     text = re.sub('<.*?>', ' ', text)
#     # remove punctuation
#     text = text.translate(str.maketrans(' ', ' ', string.punctuation))
#     # only alphabets and numerics
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     # replace newline with space
#     text = re.sub("\n", " ", text)
#     # lower case
#     text = text.lower()
#     # split and join the words
#     text = ' '.join(text.split())

#     return text












def generate_seq(lstm_model, tokenizer, seed_text, max_length, n_words):
	# print(request.form.values)
	in_text = seed_text

    # generate a fixed number of words
	for _ in range(n_words):
	        # encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		predict_x = lstm_model.predict(encoded, verbose=0)
		classes_x = np.argmax(predict_x, axis=1)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == classes_x:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text





if __name__ == "__main__":
    flask_app.run(debug=True)
