# 🚀 How to Start Patho Plant

Welcome to the **Patho Plant Agricultural Intelligence System**! 
Follow these simple steps to get the project running locally on your machine.

---

### 1️⃣ Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8+** (Recommended)
- **Git** (Optional, for version control)

---

### 2️⃣ Open the Project in Terminal
Open your terminal (Command Prompt, PowerShell, or VS Code terminal) and navigate to the project root directory:
```bash
cd "d:\Patho_Plant-master\Patho_Plant-master"
```

---

### 3️⃣ Activate the Virtual Environment
It is highly recommended to run the app within an isolated virtual environment to prevent dependency conflicts.
*(Note: A `venv` folder already exists in your directory, so you just need to activate it)*

**For Windows (PowerShell/CMD):**
```bash
.\venv\Scripts\activate
```
*(If you see `(venv)` appear at the start of your terminal line, it worked!)*

---

### 4️⃣ Install Dependencies
Once the virtual environment is activated, ensure all necessary packages (PyTorch, Flask, Leaflet, etc.) are installed by running:
```bash
pip install -r requirements.txt
```

---

### 5️⃣ Launch the Application
The core backend application is located inside the `Flask` directory. Navigate into it and run the server:
```bash
cd Flask
python app.py
```

---

### 6️⃣ Open the Dashboard
Once the terminal says `Running on http://127.0.0.1:5000/`, open your favorite web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### 🛑 Troubleshooting Common Issues
* **Permissions Error on Activation**: If Windows blocks you from running the activate script, open PowerShell as Administrator and run: `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`, then try again.
* **Missing Dependencies**: If you get a `ModuleNotFoundError` for a package like PyPDF2, you can install it manually by running: `pip install <package-name>`.
* **Geolocation & Notifications**: Ensure you click **"Allow"** when your browser prompts you for Location and Notification access, otherwise the interactive Map and Weather Alerts will remain hidden!
