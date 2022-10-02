from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about-us")
def about():
    return render_template("about-us.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/contact-us")
def contact():
    return render_template("contact-us.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    pred = {0:{'status':'Low','suggestions':[]}, 
        1:{'status':'Moderate','suggestions':[]}, 
        2:{'status':'High','suggestions':[]}}

    age = request.form['age']
    education = request.form['education']
    institution = request.form['institution']
    it = request.form['it']
    location = request.form['location']
    loadshed = request.form['loadshed']
    fincon = request.form['fincon']
    internet = request.form['internet']
    network = request.form['network']
    selflms = request.form['selflms']
    duration = request.form['duration']
    device = request.form['device']

    df = pd.DataFrame({
        'Age': {0: '21-25'},
        'Education Level': {0: 'University'},
        'Institution Type': {0: 'Non Government'},
        'IT Student': {0: 'No'},
        'Location': {0: 'Yes'},
        'Load-shedding': {0: 'Low'},
        'Financial Condition': {0: 'Mid'},
        'Internet Type': {0: 'Wifi'},
        'Network Type': {0: '4G'},
        'Class Duration': {0: '3-6'},
        'Self Lms': {0: 'No'},
        'Device': {0: 'Tab'}})
    

    col1 = df["Age"].apply(lambda x: x.split("-")[0])
    df1 = df.join(col1.to_frame(name="Lower limit Age"))
    df1.drop(['Age'], axis = 1, inplace = True)
    df1["Lower limit Age"] = df1["Lower limit Age"].astype(int)
    scaler = OrdinalEncoder()
    names = df1.columns
    d = scaler.fit_transform(df1)

    scaled_df = pd.DataFrame(d, columns=names)

    pickled_model = pickle.load(open("model.pkl", "rb"))
    y_pred = pickled_model.predict(scaled_df)

    for x in y_pred:
        return(pred[int(x)])


if __name__ == "__main__":
    app.run(debug=True)
