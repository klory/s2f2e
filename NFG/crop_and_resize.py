from PIL import Image
import numpy as np

import glob
sketch_filenames = glob.glob("data/AR_sketches/*.jpg")
sketch_filenames.sort()
neutral_filenames = glob.glob("data/AR_scream/*.bmp")
neutral_filenames.sort()

target_size = 64
neutral_width = 120
neutral_height = 165

# for i, name in enumerate(sketch_filenames):
    # sketch = Image.open(name).convert('RGB')
    # sketch = np.asarray(sketch, dtype=np.uint8)
    # sketch = sketch[50:50+neutral_height, 40:40+neutral_width]
    # sketch = Image.fromarray(sketch)
    # sketch = sketch.resize((target_size, target_size))
    # sketch.save("data/ar_s/ar_s"+str(i)+".jpg", 'JPEG')

for i, name in enumerate(neutral_filenames):
    neutral = Image.open(name).convert('RGB')
    neutral = neutral.resize((target_size, target_size))
    neutral.save("data/ar_scream/ar_scream"+str(i)+".jpg", 'JPEG')
