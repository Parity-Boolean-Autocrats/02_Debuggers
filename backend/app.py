from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from random import choice

app = Flask(__name__, static_folder="static", static_url_path="")

df = pd.read_csv(
    "https://raw.githubusercontent.com/Parity-Boolean-Autocrats/02_Debuggers/main/datasets/covid_19_in_education.csv",
    encoding="cp1252",
)
df = df.drop(["Unnamed: 9"], axis=1)
df.loc[df["gender"] == "Not Available", "gender"] = "Female"
df.loc[df["age"] == "Not Available", "age"] = "26 to 35 years old"
df.loc[
    df["geography"] == "Not Available", "geography"
] = "City center or metropolitan area"
df.loc[
    df["financial_situation"] == "Not Available", "financial_situation"
] = "I can afford food, but nothing else"


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
    lst_1 = df[df["do_children_3_and_17_yrs_receive_regular_school_meals"] == "No"]
    lst_2 = df[
        df["are_there_teachers_at_scheduled_class_hours"] == "There are not enough"
    ]
    lst_meals = lst_1[
        [
            "geography",
            "submission_state",
            "do_children_3_and_17_yrs_receive_regular_school_meals",
        ]
    ].drop_duplicates()
    lst_meals = lst_meals.to_dict(orient="index")
    lst_teachers = lst_2[
        ["geography", "submission_state", "are_there_teachers_at_scheduled_class_hours"]
    ].drop_duplicates()
    lst_teachers = lst_teachers.to_dict(orient="index")
    return render_template(
        "services.html", lst_meals=lst_meals, lst_teachers=lst_teachers
    )


@app.route("/contact-us")
def contact():
    # return render_template("form-input.html")
    return render_template("form-input.html")

@app.route("/form-submit", methods=["POST"])
def form_submit():
    print(request.form['age'])
    print(request.form['eduLevel'])
    print(request.form['institutionType'])
    print(request.form['FinancialCondtion'])
    print(request.form['Lms'])
    print(request.form['InternetType'])
    print(request.form['ITStudent'])
    print(request.form['Location'])
    print(request.form['NetworkType'])
    print(request.form['ClassDuration'])
    print(request.form['Device'])
    print(request.form['Load-Shedding'])
    return "Hell"


@app.route("/predict", methods=["GET", "POST"])
def predict():

    suggestions = {
        0: [
            "Please check whether the student is attending classes for an optimal period of time, ideally 1-3 hours, see to it whether the sessions they are attending are interactive and easy to understand.",
            "Please check whether the hardware infrastructure at your end (handheld devices, computers, etc.) is sufficient for the student to be comfortably learning online.",
            "Please try to improve the bandwidth and connectivity of your network in order to improve the quality and experience of online learning.",
        ],
        1: [
            "A little bit of assistance with online teaching, ensuring that the student is able to understand what is being taught should help in improving the student's adaptivity",
            "Try to make slight improvements in the quality of internet and networks in use to make the process of learning relatively easier for the student",
        ],
        2: ["The student is very adaptive to online learning. Well done!"],
    }

    pred = {
        0: {"status": "Low", "suggestions": choice(suggestions[0])},
        1: {"status": "Moderate", "suggestions": choice(suggestions[1])},
        2: {"status": "High", "suggestions": choice(suggestions[2])},
    }

    age = request.form["age"]
    education = request.form["eduLevel"]
    institution = request.form["institutionType"]
    it = request.form["ITStudent"]
    location = request.form["Location"]
    loadshed = request.form["Load-Shedding"]
    fincon = request.form["FinancialCondtion"]
    internet = request.form["InternetType"]
    network = request.form["NetworkType"]
    selflms = request.form["Lms"]
    duration = request.form["ClassDuration"]
    device = request.form["Device"]

    df = pd.DataFrame(
        {
            "Age": {0: age},
            "Education Level": {0: education},
            "Institution Type": {0: institution},
            "IT Student": {0: it},
            "Location": {0: location},
            "Load-shedding": {0: loadshed},
            "Financial Condition": {0: fincon},
            "Internet Type": {0: internet},
            "Network Type": {0: network},
            "Class Duration": {0: duration},
            "Self Lms": {0: selflms},
            "Device": {0: device},
        }
    )

    col1 = df["Age"].apply(lambda x: x.split("-")[0])
    df1 = df.join(col1.to_frame(name="Lower limit Age"))
    df1.drop(["Age"], axis=1, inplace=True)
    df1["Lower limit Age"] = df1["Lower limit Age"].astype(int)
    scaler = OrdinalEncoder()
    names = df1.columns
    d = scaler.fit_transform(df1)

    scaled_df = pd.DataFrame(d, columns=names)

    pickled_model = pickle.load(open("model.pkl", "rb"))
    y_pred = pickled_model.predict(scaled_df)

    output= pred[int(y_pred[0])]

    return render_template('form-input.html', output=output)
    


if __name__ == "__main__":
    app.run(debug=True)
