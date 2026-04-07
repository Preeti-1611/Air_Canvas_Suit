from flask import Flask, render_template
import subprocess
import os
import sys

app = Flask(__name__)

# Use the current python executable (works inside venv locally or on cloud providers)
VENV_PYTHON = sys.executable

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/web_canvas")
def web_canvas():
    return render_template("web_canvas.html")

@app.route("/launch_canvas")
def launch_canvas():
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(
            [VENV_PYTHON, "air_canvas.py"],
            **kwargs
        )
        return "Air Canvas launched successfully!"
    except Exception as e:
        return f"Error launching Air Canvas: {e}"

@app.route("/launch_ppt")
def launch_ppt():
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(
            [VENV_PYTHON, "pmp.py"],
            **kwargs
        )
        return "PPT Viewer launched successfully!"
    except Exception as e:
        return f"Error launching PPT Viewer: {e}"

@app.route("/launch_system")
def launch_system():
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(
            [VENV_PYTHON, "proj3a.py"],
            **kwargs
        )
        return "System Controller launched successfully!"
    except Exception as e:
        return f"Error launching System Controller: {e}"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)