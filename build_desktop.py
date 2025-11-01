import PyInstaller.__main__
import sys

PyInstaller.__main__.run([
    'desktop_app.py',
    '--name=NordlysFetcher',
    '--windowed',  # No console window
    '--onefile',   # Single executable
    '--add-data=templates:templates',  # Include HTML templates
    '--add-data=storage:storage',      # Include data directory
    '--add-data=.env:.', # Include environment file
    '--hidden-import=tiktoken_ext.openai_public',
    '--hidden-import=tiktoken_ext',
    '--collect-all=langchain',
    '--collect-all=langchain_community',
    '--collect-all=chromadb',
    '--icon=icon.ico',  # Optional: Add your icon
])