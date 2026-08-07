import streamlit as st
import os
import sys

# Set page config ONCE for the entire app
st.set_page_config(
    page_title="Detection Engine",
    layout="wide"
)

# Prevent sub-apps from calling set_page_config again
st.set_page_config = lambda *args, **kwargs: None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sidebar navigation
st.sidebar.title("Detection Engine")
st.sidebar.markdown("---")

task = st.sidebar.radio(
    "Select Task:",
    [
        "Task 1 - Car Colour Detection",
        "Task 2 - Animal Detection",
        "Task 3 - Voice Emotion Detection",
        "Task 4 - Nationality Detection",
        "Task 5 - ASL Sign Language",
        "Task 6 - Drowsiness Detection"
    ]
)

# Extract task number and build path
task_num = task.split("-")[0].strip().split()[-1]
task_dir = os.path.join(ROOT_DIR, f"task_{task_num}")
app_path = os.path.join(task_dir, "app.py")

# Ensure task directory is in sys.path
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

# Change working directory so relative paths in sub-apps work
os.chdir(task_dir)

# Read and execute the selected task's app.py
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, app_path, "exec"), {"__file__": app_path, "__name__": "__main__"})
