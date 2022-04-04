# Pokemon Card Recognizer

Recognize a Pokemon Card in an image or video.

```python
from card_recognizer.api.card_recognizer import CardRecognizer, Mode 
recognizer = CardRecognizer(
    mode=Mode.PULLS_VIDEO
)
pulls = recognizer.exec("/path/to/video")
```
![](https://github.com/prateekt/pokemon-card-recognizer/blob/16ac467082080230a0da9e3667d896951c3db681/example.png?raw=true)

Benchmarks: https://docs.google.com/presentation/d/1nJKkQicFS1gC5c3fRBNyJIRX14930S1OZKWBiA9YtVw/edit?usp=sharing


    
