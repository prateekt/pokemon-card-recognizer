# Pokemon Card Recognizer

Recognize a Pokemon Card in an image or video.

```python
from card_recognizer.api.card_recognizer import CardRecognizer, Mode 
recognizer = CardRecognizer(
    mode=Mode.PULLS_VIDEO
)
pulls = recognizer.exec("/path/to/video")
```

Example analysis of a booster pack opening video:

![](https://github.com/prateekt/pokemon-card-recognizer/blob/75409e8ecdc32256dfc4a0a8243782152fdd406b/example2.png?raw=true)
![](https://github.com/prateekt/pokemon-card-recognizer/blob/75409e8ecdc32256dfc4a0a8243782152fdd406b/example.png?raw=true)

Benchmarks: https://docs.google.com/presentation/d/14JzyJ8jWJb5JutFXDNDciMFHf8QLCr7Qfpk2S9fvK-s/edit?usp=sharing


    
