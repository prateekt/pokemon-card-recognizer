# Pokemon Card Recognizer

Recognize a Pokémon Card in an image or video.

```python
from card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode

# init and set output paths
recognizer = CardRecognizer(
    mode=OperatingMode.PULLS_VIDEO
)
recognizer.set_summary_file(summary_file="summary.txt")
recognizer.set_output_path(output_path="out_figs")

# run recognizer on video and visualize results
pulls = recognizer.exec("/path/to/video")
recognizer.vis()
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

Note that the CardRecognizer works MUCH (~5x-10x) faster on NVIDIA GPU, so it is highly recommended that you have CUDA. If CUDA is available, the CardRecognizer will automatically use it. Otherwise, the CardRecognizer will default to CPU which is substantially slower and not recommended for batch processing.

If processing video files, you may also need to download and install FFMPEG (https://ffmpeg.org/) which is used to uncompress videos into image frames.

On linux:
```commandline
sudo apt update
sudo apt install ffmpeg
```

Optional: The default OCR backend used by CardRecognizer is easy_ocr which is installed via the conda, but if you choose to use PyTesseract instead, you will need to install it:
```commandline
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev  
```

<b>Example Usage to Recognize a Card in a Single Image:</b>

```python
from card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode
recognizer = CardRecognizer(
    mode=OperatingMode.SINGLE_IMAGE,
    set_name="master"
)
pred_result = recognizer.exec("/path/to/image")
detected_card = recognizer.classifier.reference.lookup_card_prediction(
    card_prediction=pred_result[0]
)
print(detected_card.set)
print(detected_card.name)
print(detected_card.number)
```
<b>Useful Operating Modes for CardRecognizer: </b>

```commandline
# process a single image
OperatingMode.SINGLE_IMAGE

# process a directory of images
OperatingMode.IMAGE_DIR

# process a video file
OperatingMode.VIDEO

# process a video file where cards are being shown ("pulled") sequentially 
(no assumption on # of cards shown in the video)
OperatingMode.PULLS_VIDEO

# process a video file where cards are being shown sequentially, 
# coming from a booster pack (assumes 11 cards are being shown in the video and 
# finds the best statistical estimation of at most 11 most likely shown cards).
OperatingMode.BOOSTER_PULLS_VIDEO
```
<b> Card Reference Sets </b>

Note that you can change "set_name" in the CardRecognizer constructor to whatever specific set reference (e.g. "base1", "jungle", etc) you want. For list of supported reference sets, view
```python
from card_recognizer.reference.core.build import ReferenceBuild
print(ReferenceBuild.supported_card_sets())
```
<b>Building Card Reference:</b>

The pypi package and GitHub source for pokemon-card-recognizer comes bundled with pre-rebuilt references for all major Pokémon card sets. It is recommended to use the pre-built references. If however, you want to rebuild the reference for some reason, you can do:

```commandline
python card_recognizer/reference/core/build.py [PTCGSDK_API_KEY]
```
where `PTCGSDK_API_KEY` is your PTCGSDK API key. You can get one here: https://pokemontcg.io/.

Note that building and evaluating the reference can take an hour or more, depending on your system configuration. It is recommended to use the prebuilt card references that come pre-bundled.

<b> *This API is unofficial, open-source, in development, and not affiliated with the Pokémon Company. </b>