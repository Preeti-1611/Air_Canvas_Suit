from flask import Flask, render_template
import subprocess
import os

app = Flask(__name__)

# Path to venv python
VENV_PYTHON = os.path.join("venv", "Scripts", "python.exe")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/launch_canvas")
def launch_canvas():
    try:
        subprocess.Popen(
            [VENV_PYTHON, "air_canvas.py"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return "Air Canvas launched successfully!"
    except Exception as e:
        return f"Error launching Air Canvas: {e}"

@app.route("/launch_ppt")
def launch_ppt():
    try:
        subprocess.Popen(
            [VENV_PYTHON, "pmp.py"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return "PPT Viewer launched successfully!"
    except Exception as e:
        return f"Error launching PPT Viewer: {e}"

@app.route("/launch_system")
def launch_system():
    try:
        subprocess.Popen(
            [VENV_PYTHON, "proj3a.py"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return "System Controller launched successfully!"
    except Exception as e:
        return f"Error launching System Controller: {e}"

if __name__ == "__main__":
    app.run(debug=True)