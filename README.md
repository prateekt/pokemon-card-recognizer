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

![](https://github.com/prateekt/pokemon-card-recognizer/blob/main/example2.png?raw=true)
![](https://github.com/prateekt/pokemon-card-recognizer/blob/16ac467082080230a0da9e3667d896951c3db681/example.png?raw=true)

Benchmarks: https://docs.google.com/presentation/d/10--ByPMkb6OnwhdJqrPBOhMqMDLk3pliL27znlQ_jUo/edit?usp=sharing


    
