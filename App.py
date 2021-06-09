from flask import Flask, render_template, request, redirect, url_for, flash
from joblib.numpy_pickle import load
from  flaskext.mysql import MySQL
from joblib import dump, load
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


app=Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_DB'] = 'Flask_Deploy'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASS'] = ''
mysql=MySQL(app)

# Sesion 
app.secret_key='mysecretkey'

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predecir', methods=["POST"])
def predecir():
    if request.method=='POST':
        TestTweet= [request.form['TestTweet']]
        nb=load('models/NaiveBayes.pkl')
        vec=load('models/tfidf_vect.pkl')
        valorTrans=vec.transform(TestTweet)
        prediccion=nb.predict(valorTrans)
      
        flash([str(TestTweet[0]), str(prediccion[0])])
        return redirect(url_for('index'))
       


if __name__=='__main__':
    app.run(port=3000, debug=True)
