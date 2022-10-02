from flask import Flask, render_template, request

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


if __name__ == "__main__":
    app.run(debug=True)
