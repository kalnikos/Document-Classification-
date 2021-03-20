import flask
import pickle
from flask import Flask,render_template,url_for,request
from sklearn import svm
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from pathlib import Path
import docx2txt
import os

UPLOAD_FOLDER = "/home/ubuntu"


app = flask.Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#from flask_cors import CORS
#CORS(app)

@app.route('/')
def home():
    return render_template('home.html')


def text_pre(text):
    ## text preprocessing 
    import nltk 
    nltk.download('punkt') 
    nltk.download('averaged_perceptron_tagger') 
    nltk.download('wordnet') 
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer() 
    from nltk.corpus import stopwords 
    nltk.download('stopwords') 
    stop_words = set(stopwords.words('english')) 
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    
    text = text.lower() 
    temp_sent =[] 
    words = nltk.word_tokenize(text) 
    tags = nltk.pos_tag(words) 
    
    for i, word in enumerate(words): 
        if tags[i][1] in VERB_CODES:   
            lemmatized = lemmatizer.lemmatize(word, 'v') 
        else: 
            lemmatized = lemmatizer.lemmatize(word) 
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized) 

    finalsent = ' '.join(temp_sent) 
    finalsent = finalsent.replace("n't", " not") 
    finalsent = finalsent.replace("'m", " am") 
    finalsent = finalsent.replace("'s", " is") 
    finalsent = finalsent.replace("'re", " are") 
    finalsent = finalsent.replace("'ll", " will") 
    finalsent = finalsent.replace("'ve", " have") 
    finalsent = finalsent.replace("'d", " would") 
    
    ## Load the vectorizer model
    path_model = "vectorizer.pickle"
    with open(path_model, 'rb') as dt:
         vectorizer = pickle.load(dt)
            
    ## Apply the model to the text document
    finalsent = [finalsent]
    finalsent = vectorizer.transform(finalsent).toarray()
    
    ## Apply PCA to the uploaded document
    path = "pca.pickle"
    with open(path, 'rb') as dt:
         pca = pickle.load(dt)
            
    ## pca  model
    finalsent = pca.transform(finalsent)

    return finalsent 
        
path_model = "best_svm.pickle"
with open(path_model, 'rb') as dt:
        svm_model = pickle.load(dt)

@app.route('/predict',methods=['POST'])
def predict():         
    if request.method == 'POST':
        message = request.form['message']
        message = text_pre(message)
        if svm_model.predict_proba(message).max(axis=1) > 0.65:
                       data = svm_model.predict(message)[0]
                       return render_template('result.html', prediction = data)
        else:
                data = 8
                return render_template('result.html', prediction = data)



@app.route('/predic',methods=['GET', 'POST'])
def predic():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = Path(filename).absolute()
        my_new =  docx2txt.process(path)
        message = text_pre(my_new)
        if svm_model.predict_proba(message).max(axis=1) > 0.65:
                       data = svm_model.predict(message)[0]
                       return render_template('result.html', pred = data)
        else:
             data = 8
             return render_template('result.html', pred = data)


if __name__ == '__main__':
      app.run(host='0.0.0', port=8080)