# Quality Feature Extraction
A pipeline for extracting image quality features for use in machine learning tasks.  A number of quality features are extracted such as light, color, rule of thirds, texture, smoothness, blurriness, depth of field, and scene composition.

# How to use
```python
USAGE:
    blind_features.py (-d <dir> | <image>...)
```
Pass a list of images or a directory to extract the features.  The output will be a JSON array of objects.  Each object being the extracted features for an image.
