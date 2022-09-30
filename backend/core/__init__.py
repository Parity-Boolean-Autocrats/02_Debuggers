from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv


def create_app():

    app = Flask(__name__)

    CORS(app)
    load_dotenv()

    from core.general.routes import general

    app.register_blueprint(general)

    return app
