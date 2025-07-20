# NoMoreShrimp
** No More Shrimp ** is a real-time posture monitoring application we built as the HackThe6ix project. It helps users improve their sitting posture
while working by analyzing webcam input and giving instant, actionable feedback. No more hunching like a shrimp at your desk. Sit tall and stay healthy !!

## Features
- Real-time posture analysis using **Mediapipe Pose**
- Tracks **33 body landmarks** with high accuracy, even during subtle movements
- Detects poor posture indicators like:
  - Forward head tilt
  - Slouching back
  - Uneven shoulders
- Adaptive scoring for **front, side, and back camera angles**
- Clean and detailed **Heads-Up Display (HUD)** with live feedback
- Saves session data as **JSON files with metadata** to track improvements over time
- Runs locally, privacy-friendly, no cloud required

## Built With
- **Python**
- [Mediapipe](https://google.github.io/mediapipe/) — real-time body landmark detection
- [OpenCV](https://opencv.org/) — webcam input and visualization
- **NumPy** — numerical calculations
- **JSON** — save and load session states with metadata

## What We Learned
While building No More Shrimp, we learned how to:
- Use Mediapipe and OpenCV together for real-time computer vision
- Extract and process body landmarks data effectively
- Design a UI that's clear, informativ,e and adaptive to different viewing angles
- Save and load structured session data with helpful metadata using JSON

## What's Next
We're excited to keep improving No More Shrimp:
- Better UI and UX for a more polished experience
- Improved side posture detection and analysis
- Progress tracking and smarter feedback over time
- Tips and tricks for users on how to improve posture

## How to Run
1. Clone this repository:
bash
   git clone https://github.com/yourusername/no-more-shrimp.git
   cd no-more-shrimp
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python main.py


