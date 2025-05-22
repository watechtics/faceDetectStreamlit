# Face Detection App

A Streamlit web application for real-time face detection using OpenCV.

## Features
- Upload and detect faces in images
- Process videos for face detection
- User-friendly web interface
- Download processed results

## How to Use
1. Visit the deployed app
2. Upload an image or video
3. View the face detection results
4. Download the processed file

## Technologies Used
- Streamlit
- OpenCV
- Python
- Computer Vision

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py

### 3. Create `.streamlit/config.toml` (Optional)

Create a folder `.streamlit` and inside it create `config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
