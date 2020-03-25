#import libraries
import sys
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger']) # download for lemmatization
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine

#Define load function
def load_data(database_filepath):

    """Load and merge messages and categories datasets
    
    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.
       
    Returns:
    X: dataframe. Dataframe containing features dataset.
    Y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    
    #create engine: engine
    engine = create_engine('sqlite:///' + database_filepath)
    # Load data from database

    df = pd.read_sql_table('df',con=engine) 
    # Create X and Y datasets
    
    X = df.iloc[:,1] 
    Y = df.iloc[:,4:] 
    category_names=list(df.columns[4:])

    return X,Y,category_names

def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation and splits into words

    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]"," ",text) 
    words=word_tokenize(text) 
    
    stop_words = stopwords.words("english")

    #words = [w for w in words if w not in stopwords.words("english")]

    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    return words


def build_model():
    """Build a machine learning pipeline
    
    Args:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)), #create the CountVectorizer object
        ('tfidf', TfidfTransformer()), #create Tfidftransformer object    
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC()))) #create the Classifier object
    ])

    # Create parameters dictionary
    parameters = {   
        'clf__estimator__estimator__C': [1],
        'tfidf__use_idf': [False],
        'vectorizer__max_df': [0.8],
        'vectorizer__ngram_range':(1, 1)
    }

    parameters={}
    #create a grid searchCV for clarity of code
    
    grid_cv = GridSearchCV(pipeline, param_grid=parameters, cv=5,verbose=3)
    return grid_cv

def evaluate_model(model, X_test, Y_test, category_names):

    """Returns test accuracy, precision, recall and F1 score for fitted model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """

    Y_pred = model.predict(X_test)

    Y_test=pd.DataFrame(data=Y_test,columns=category_names) #Convert prediction numpy into dataframe
    Y_pred=pd.DataFrame(data=Y_pred,columns=category_names)
    
    for column in Y_pred.columns:
        print(column)
        print(classification_report(Y_test[column], Y_pred[column]))
        print('_____________________________________________________')


def save_model(model, model_filepath):

    """Pickle fitted model
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """ 

    pickle_out = open(model_filepath,'wb')
    pickle.dump(model, pickle_out)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()