import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QUrl, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from threading import Thread
import time

# Import your Flask app
from app import app as flask_app

class DesktopRAGApp(QWebEngineView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nordlys Fetcher - Desktop Edition")
        self.resize(1400, 900)
        
        # Ensure storage directories exist
        Path("./storage/chroma").mkdir(parents=True, exist_ok=True)
        Path("./storage/uploads").mkdir(parents=True, exist_ok=True)
        
        # Start Flask server in background thread
        self.flask_thread = Thread(target=self.run_flask, daemon=True)
        self.flask_thread.start()
        
        # Wait for Flask to start, then load UI
        QTimer.singleShot(1500, self.load_ui)
        
    def run_flask(self):
        """Run Flask server in background"""
        try:
            flask_app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask server error: {e}")
    
    def load_ui(self):
        """Load the Flask app in the window"""
        self.load(QUrl("http://127.0.0.1:8000"))
    
    def closeEvent(self, event):
        """Handle window close"""
        reply = QMessageBox.question(
            self, 
            'Exit Application',
            'Are you sure you want to close Nordlys Fetcher?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Nordlys Fetcher")
    
    # Check if .env exists
    if not Path(".env").exists():
        QMessageBox.critical(
            None,
            "Configuration Missing",
            "Please create a .env file with your API keys before running the application."
        )
        sys.exit(1)
    
    window = DesktopRAGApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()