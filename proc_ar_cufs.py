from PIL import Image
import numpy as np
import glob
import os
from shutil import rmtree

sketch_width = 200
sketch_height = 250
neutral_width = 120
neutral_height = 165

target_size = 128

sketch_folder_path = './data/' + str(target_size) + '/sketch/'
if os.access(sketch_folder_path, os.F_OK):
    rmtree(sketch_folder_path)
os.mkdir(sketch_folder_path)

sketch_filenames = glob.glob("./data/My AR/sketch/*.jpg")
sketch_filenames.sort()
for i, name in enumerate(sketch_filenames):
    sketch = Image.open(name).convert('RGB')
    sketch = np.asarray(sketch, dtype=np.uint8)
    sketch = sketch[50:50 + neutral_height, 40:40 + neutral_width]
    sketch = Image.fromarray(sketch)
    sketch = sketch.resize((target_size, target_size))
    sketch.save(sketch_folder_path + str(i) + ".jpg", 'JPEG')


photo_folders = ['neutral/', 'smile/', 'anger/', 'scream/']

for folder in photo_folders:
    photo_folder_path = './data/' + str(target_size) + '/' + folder
    if os.access(photo_folder_path, os.F_OK):
        rmtree(photo_folder_path)
    os.mkdir(photo_folder_path)

    neutral_filenames = glob.glob("./data/My AR/" + folder + "/*.bmp")
    neutral_filenames.sort()
    for i, name in enumerate(neutral_filenames):
        neutral = Image.open(name).convert('RGB')
        neutral = neutral.resize((target_size, target_size))
        neutral.save(photo_folder_path + str(i) + ".jpg", 'JPEG')
