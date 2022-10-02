from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

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
    return render_template("contact-us.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    pred = {
        0: {"status": "Low", "suggestions": ["Test"]},
        1: {"status": "Moderate", "suggestions": ["Test"]},
        2: {"status": "High", "suggestions": ["Test"]},
    }

    age = request.form["age"]
    education = request.form["education"]
    institution = request.form["institution"]
    it = request.form["it"]
    location = request.form["location"]
    loadshed = request.form["loadshed"]
    fincon = request.form["fincon"]
    internet = request.form["internet"]
    network = request.form["network"]
    selflms = request.form["selflms"]
    duration = request.form["duration"]
    device = request.form["device"]

    df = pd.DataFrame(
        {
            "Age": {0: "21-25"},
            "Education Level": {0: "University"},
            "Institution Type": {0: "Non Government"},
            "IT Student": {0: "No"},
            "Location": {0: "Yes"},
            "Load-shedding": {0: "Low"},
            "Financial Condition": {0: "Mid"},
            "Internet Type": {0: "Wifi"},
            "Network Type": {0: "4G"},
            "Class Duration": {0: "3-6"},
            "Self Lms": {0: "No"},
            "Device": {0: "Tab"},
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

    for x in y_pred:
        return pred[int(x)]


if __name__ == "__main__":
    app.run(debug=True)
