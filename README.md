# smile-o-meter
Smile measuring device with bargraph LED display for Raspberry Pi

## The Problem
Lots of video calls lately, and everyone looking at eachothers faces.  But for some people (e.g. me), the thinking/concentrating face can look quite grumpy.  RBF is another acronymn sometimes used here.  Overall, I want to retrain myself to be cheerful on calls.

## The solution
A pre-trained mediapipe model https://google.github.io/mediapipe/solutions/face_mesh.html is used to generate a continous face mesh from a video feed.  This is fed into a  machine-learning model, set up to perform a linear regression to determine the closest smile index (1-9) to the face mesh provided.  This face mesh is then displayed on the bar graph LED.  The raspberry pi is then set up to run this script at startup.

In addition a button is provided - if held down for >3seconds, it enters a training mode, where the user poses 9 times to collect new data, instructed by flashing the various LEDs.  It is up to the user to decide how sad/happy the extremes are.  At the end the machine learning algorithm does its retraining and starts operating with the new model.  This model is saved to the SD card ready for the next time it is run.

Overall you get a small box with a camera on it and some LEDs which tells you how smiley you are.  It is silent and doesn't need any internet connection.


## Hardware needed
- Raspberry Pi (I used RPi4, which seems just powerful enough, so wouldn't recommend any lower).
- Raspberry Pi camera rev1.3 (the cheap one) connected by ribbon cable.
- A custom made piece of stripboard with a hole cut in it for the camera.
- A dual row connector
- A bargraph LED chip (Red,Yellow,Green,Blue)
- Raspberry pi case to mount it in.

Please take this document as a guide and general description rather than a step-by-step instruction.

## Future
Possible future additions to this:
- Share the smile index via MQTT or similar and track it (could be used during the day, not necessarily at meetings).
