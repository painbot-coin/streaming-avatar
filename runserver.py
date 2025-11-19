import os
from src import app


if __name__ == "__main__":
    HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    ssl_context=('fullchain.pem', 'privkey.pem')
    try:
        PORT = int(os.environ.get("SERVER_PORT", "443"))
    except ValueError:
        PORT = 1234
    app.secret_key = "1cd6f35db029d4b8fc98fc05c9efd06a2e2cd1ffc3774d3f035ebd8d"
    app.config['SERVER_NAME'] = "avatar.facefile.co"
    app.run(HOST, PORT, ssl_context=ssl_context, debug=False, threaded=True)
