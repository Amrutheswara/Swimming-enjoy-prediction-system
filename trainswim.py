import pickle
import numpy as np
from flask import Flask,render_template,request

app=Flask(__name__)

@app.route('/',methods=["POST","GET"])
def index():
    if request.method=='POST':
        sky=request.form['sky']

        testdata=[]
        for item in request.form:
            testdata.append(float(request.form[item]))
        print(testdata)

        file=open('knnswim.pkl','rb')
        model=pickle.load(file)

        td=np.array([testdata])

        pred=model.predict(td)

        if pred[0]==1:
            msg="you can enjoy the swimming"
        else:
            msg="sorry you can't enjoy the swimming"
        print(msg)
        return render_template('swim.html',res=msg)
    else:
        return render_template('swim.html')

if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0")

