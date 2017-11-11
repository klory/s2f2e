import tensorflow as tf
from PIL import Image
import numpy as np

import glob
sketch_filenames = glob.glob("small_data/AR_sketches/*.jpg")
sketch_filenames.sort()
neutral_filenames = glob.glob("small_data/AR_neutral/*.bmp")
neutral_filenames.sort()

for i, name in enumerate(sketch_filenames):
    sketch = Image.open(name).convert('RGB')
    sketch = np.asarray(sketch, dtype=np.uint8)
    sketch = sketch[50:215, 40:160]
    sketch = Image.fromarray(sketch)
    sketch = sketch.resize((128, 128))
    sketch.save("small_data/ar_s/ar_s"+str(i)+".jpg", 'JPEG')

for i, name in enumerate(neutral_filenames):
    neutral = Image.open(name).convert('RGB')
    neutral = neutral.resize((128, 128))
    neutral.save("small_data/ar_nf/ar_nf"+str(i)+".jpg", 'JPEG')
