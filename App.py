from flask import Flask, render_template, request
from joblib.numpy_pickle import load
from flaskext.mysql import MySQL
from joblib import dump, load


app=Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_DB'] = 'Flask_Deploy'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASS'] = ''
mysql=MySQL(app)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predecir', methods=["POST"])
def predecir():
    if request.method=='POST':
        Valores=[[float(request.form['val1']), float(request.form['val2']),float(request.form['val3']),float(request.form['val4']),float(request.form['val5']),
        float(request.form['val6']),float(request.form['val7']),float(request.form['val8']),float(request.form['val9']), float(request.form['val10']), float(request.form['val11'])]]
        #suma=Valores[0] + Valores[1]


    # json=request.get_json(force=True)
    # Valores=json['Valores']
        clf=load('models/Casif_vinos_Binario1.pkl')
        prediccion=clf.predict(Valores)
        return "prediccion\n"+str(prediccion[0])+"\n"


if __name__=='__main__':
    app.run(port=3000, debug=True)
