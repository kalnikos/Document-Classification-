import flask
import pickle
from flask import Flask,render_template,url_for,request
from sklearn import svm

app = flask.Flask(__name__)

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
    path_model = r"C:\Users\nikos\Desktop\Classify reviews\Pickles\vectorizer.pickle"
    with open(path_model, 'rb') as dt:
         vectorizer = pickle.load(dt)
            
    ## Apply the model to the text document
    finalsent = [finalsent]
    finalsent = vectorizer.transform(finalsent).toarray()
    
    ## Apply PCA to the uploaded document
    path = r"C:\Users\nikos\Desktop\Classify reviews\model_data\pca.pickle"
    with open(path, 'rb') as dt:
         pca = pickle.load(dt)
            
    ## pca  model
    finalsent = pca.transform(finalsent)

    return finalsent 

def category_name(result):
    encode = {0:'Admin', 1:'Analyst', 2:'cleaner', 3:'Finance', 4:'hospitality',
       5:'recruitment', 6:'Warehouse', 7:'web developer'}
    
    if result in encode:
        print(encode[result])
    else:
        print("I dont have the answer")
        
@app.route('/predict',methods=['POST'])
def predict():
    path_model = r"C:\Users\nikos\Desktop\models_df\best_svm.pickle"
    with open(path_model, 'rb') as dt:
         svm_model = pickle.load(dt)
    
    #import docx2txt
    #myfile = r"C:\Users\nikos\Desktop\web_scraping\monster-cv-template-admin-assistant.docx"
    #finalsent =  docx2txt.process(myfile)
    if request.method == 'POST':
        #finalsent = request.args["Document"]
        message = request.form['message']
        data = svm_model.predict(text_pre(message))
    #   text_pred = category_name(text_pred)
        return render_template('result.html', prediction = data)


if __name__ == '__main__':
    app.run(debug=True)