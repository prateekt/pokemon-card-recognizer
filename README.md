# Pokemon Card Recognizer

Recognize a Pokemon Card in an image or video.

```python
from card_recognizer.api.card_recognizer import CardRecognizer 
recognizer = CardRecognizer(
    mode=Mode.PULLS_VIDEO
)
pulls = recognizer.exec("/path/to/video")
```
![Alt text](example.png)

Benchmarks: https://docs.google.com/presentation/d/1nJKkQicFS1gC5c3fRBNyJIRX14930S1OZKWBiA9YtVw/edit?usp=sharing


    
