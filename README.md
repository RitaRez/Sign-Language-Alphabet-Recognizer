# Sign Language Alphabet Recognizer
Through a flask application the program access the webcam conected to the device being used, takes pictures of the user and predicts which sign language alphabet letter is being displayed. Works waay better on a white background.


## To run virtual environment:

```bash
pip install virtualenv
virtualenv Videorecognition
source Videorecognition/bin/activate
```

## To install project dependencies:

```bash
sudo apt install python3-opencv
pip install -r requirements.txt
```

## To run flask app:

```bash
cd Videorecognition
export FLASK_APP=webstreaming.py
flask run
```
