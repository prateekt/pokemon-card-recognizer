# Pokemon Card Recognizer

Recognize a Pokemon Card in an image.

Pokemon Card Recognizer is a framework for identifying Pokemon cards in images based.

```python
recognizer = CardRecognizerPipeline(
    set_name="Brilliant Stars", mode=Mode.PULLS_VIDEO
)
pulls = recognizer.exec("/path/to/video")
print(pulls)
```

Benchmarks: https://docs.google.com/presentation/d/1nJKkQicFS1gC5c3fRBNyJIRX14930S1OZKWBiA9YtVw/edit?usp=sharing


    
