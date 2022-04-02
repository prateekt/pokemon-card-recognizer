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



    
