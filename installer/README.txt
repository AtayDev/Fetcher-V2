===========================================
   NORDLYS FETCHER - DESKTOP EDITION
   Version 1.0 | Document RAG System
===========================================

QUICK START GUIDE
=================

STEP 1: Get Your FREE API Key (5 minutes)
------------------------------------------
1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with "AIza...")
4. Keep it ready for setup


STEP 2: First Launch Setup
---------------------------
1. Launch "Nordlys Fetcher" from Start Menu or Desktop
2. On first run, the app will create a config file
3. When Notepad opens automatically:
   - Find the line: GOOGLE_API_KEY=your_api_key_here
   - Replace "your_api_key_here" with your actual key
   - Save and close Notepad (Ctrl+S)
4. Restart the application


STEP 3: Upload Documents
-------------------------
1. Click "Dashboard" in the sidebar
2. Click "Browse Files" or drag & drop files
3. Supported formats: PDF, DOCX, TXT
4. Click "Index Document"
5. Wait for confirmation message


STEP 4: Ask Questions
---------------------
1. Click "Chat Console" in the sidebar
2. Type your question about the documents
3. Press Enter or click the send button
4. AI searches your documents and responds with sources


FEATURES
========
✅ All documents stored locally on your computer
✅ Intelligent semantic search across all your files
✅ Source citations with page numbers (PDF files)
✅ Support for multiple document formats
✅ Privacy-focused: only queries sent to API, not documents


TROUBLESHOOTING
===============

❌ App won't start:
   → Check .env file exists in installation folder
   → Verify API key has no extra spaces or quotes
   → Right-click app → "Run as administrator"

❌ "Configuration Missing" error:
   → Make sure you added your API key to .env file
   → Restart the application after editing .env

❌ "No relevant information found":
   → Ensure you've uploaded documents first
   → Check documents are successfully indexed (Dashboard)
   → Try rephrasing your question

❌ Windows SmartScreen warning:
   → Click "More info"
   → Click "Run anyway"
   → This is normal for new applications


WHERE IS MY DATA?
=================
Installation folder: C:\Program Files\Nordlys Fetcher\
Documents: C:\Program Files\Nordlys Fetcher\storage\uploads\
Database: C:\Program Files\Nordlys Fetcher\storage\chroma\
Config: C:\Program Files\Nordlys Fetcher\.env


PRIVACY & SECURITY
==================
✅ All files stay on your computer
✅ No document uploads to cloud
✅ Only your questions are sent to Gemini API
✅ Responses come back from API
✅ No telemetry or tracking


UNINSTALLING
============
Control Panel → Programs → Uninstall a program → Nordlys Fetcher


SUPPORT
=======
For issues or questions:
Email: your_email@example.com
GitHub: https://github.com/yourusername/nordlys-fetcher


LICENSE
=======
Copyright © 2025 Your Company
Licensed under MIT License


VERSION HISTORY
===============
v1.0 (2025-01-XX)
- Initial release
- PDF, DOCX, TXT support
- Gemini API integration
- Multi-document search