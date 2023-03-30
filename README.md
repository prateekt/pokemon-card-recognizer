# Pokemon Card Recognizer

Recognize a Pokemon Card in an image or video.

```python
from card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode

recognizer = CardRecognizer(
    mode=OperatingMode.PULLS_VIDEO
)
pulls = recognizer.exec("/path/to/video")
```

Example analysis of a booster pack opening video:

![](https://github.com/prateekt/pokemon-card-recognizer/blob/75409e8ecdc32256dfc4a0a8243782152fdd406b/example2.png?raw=true)
![](https://github.com/prateekt/pokemon-card-recognizer/blob/75409e8ecdc32256dfc4a0a8243782152fdd406b/example.png?raw=true)

<b>Benchmarks</b>: https://docs.google.com/presentation/d/1Q6PzJpqyyLFvtkLGoCFeXaHYOauOzeF_umx8V3w4fV8/edit?usp=sharing

<b>Installation:</b>

```
make conda_dev
conda activate card_rec_env
pip install pokemon-card-recognizer
```    

<b>Building Card Reference:</b>

You can use the prebuilt card references for various Pokemon card sets or build it yourself using
```commandline
python card_recognizer/reference/core/build.py [PTCGSDK_API_KEY]
```
where `PTCGSDK_API_KEY` is your PTCGSDK API key. You can get one here: https://pokemontcg.io/.

Note that building the reference can take an hour or more, depending on your system configuration.
