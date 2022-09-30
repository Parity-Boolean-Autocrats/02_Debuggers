from flask import Blueprint, jsonify
from config import BASE_URL

general = Blueprint("general", __name__)

# Routes


@general.route(f"{BASE_URL}/")
def index():
    return jsonify({"greeting": "Welcome to the API!"})
