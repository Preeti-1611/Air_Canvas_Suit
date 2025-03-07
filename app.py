from flask import Flask, render_template
import subprocess
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Your HTML file

@app.route("/launch_canvas")
def launch_canvas():
    try:
        subprocess.Popen([sys.executable, "air_canvas.py"])  # Launch the Air Canvas script
        return "Air Canvas launched successfully!"
    except Exception as e:
        return f"Error launching Air Canvas: {e}"

@app.route("/launch_ppt")
def launch_ppt():
    try:
        subprocess.Popen([sys.executable, "pmp.py"])  # Launch the PPT Viewer script
        return "PPT Viewer launched successfully!"
    except Exception as e:
        return f"Error launching PPT Viewer: {e}"
    
@app.route("/launch_system")
def launch_system():
    try:
        subprocess.Popen([sys.executable, "proj3a.py"])  # Launch the PPT Viewer script
        return "PPT Viewer launched successfully!"
    except Exception as e:
        return f"Error launching PPT Viewer: {e}"

if __name__ == "__main__":
    app.run(debug=True)
