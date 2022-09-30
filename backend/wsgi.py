from core import create_app
from config import get_config

app = create_app()

# ? *** Driver function ***
if __name__ == "__main__":
    app.run(debug=get_config("FLASK_ENV"))