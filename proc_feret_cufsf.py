import glob
import os
from shutil import rmtree
from PIL import Image, ImageOps
import re
from pathlib import Path
import bz2

target_size = 128

sketch_names = glob.glob("./data/CUFSF/original_sketch/*.jpg")
sketch_names.sort()

photo_names = glob.glob("./data/My FERET/*.ppm.bz2")
photo_names.sort()

photo_link_file = "./data/CUFSF/names_of_photos_used_for_drawing_sketches.txt"
f = open(photo_link_file, 'r')
photos_names = f.readlines()
photos_names = map(str.strip, photos_names)

sketch_path = "./data/" + str(target_size) + "/feret_s"
neutral_path = ".data/" + str(target_size) + "/feret_nf"
if os.access(sketch_path, os.F_OK):
    rmtree(sketch_path)
if os.access(neutral_path, os.F_OK):
    rmtree(neutral_path)
os.mkdir(sketch_path)
os.mkdir(neutral_path)


def crop_from_fiducial_file(image, fiducial_file):
    width, height = image.size

    f = open(fiducial_file)
    lines = f.readlines()
    f.close()
    points = []
    for p in lines:
        p = p.strip()
        p = p.split(' ')
        p = list(map(float, p))
        points.append(p)

    pupil_dist = points[1][0] - points[0][0]
    eye_to_mouth_dist = points[2][1] - points[0][1]
    left = max(0, points[0][0] - pupil_dist / 1.5)
    up = max(0, points[0][1] - eye_to_mouth_dist * 1.5)
    right = min(width, points[1][0] + pupil_dist / 1.5)
    down = min(height, points[2][1] + eye_to_mouth_dist / 1.5)
    image_cropped = image.crop((left, up, right, down))
    return image_cropped


for sketch_name, photo_name in zip(sketch_names, photos_names):
    m = re.findall('\d+', photo_name)
    photo_idx = m[0]
    photo_date = m[-1]
    m = re.findall('[a-z]+', photo_name)
    photo_pose = m[0]

    filepath = ''.join(["./data/My FERET/", photo_idx, '.ppm.bz2'])
    my_file = Path(filepath)
    if my_file.is_file():
        with bz2.BZ2File(filepath) as file:
            photo = Image.open(file).convert('RGB')
            photo_with_border = ImageOps.expand(
                photo, border=(100, 0), fill='black')
            # get photo points
            name = photo_name.split('/')[-1].split('.')[0]
            path = os.path.join('./data/CUFSF/photo_points',
                                '.'.join([name, '3pts']))
            photo_cropped = crop_from_fiducial_file(photo_with_border, path)
            photo_cropped = photo_cropped.resize((target_size, target_size))
            photo_cropped.save(os.path.join(
                neutral_path, sketch_name.split('/')[-1]))

            sketch = Image.open(sketch_name).convert('RGB')
            print(sketch.size)
            # get sketch points
            path = os.path.join('./data/CUFSF/sketch_points',
                                '.'.join([photo_idx, '3pts']))
            sketch_cropped = crop_from_fiducial_file(sketch, path)
            sketch_cropped = sketch_cropped.resize((target_size, target_size))
            sketch_cropped.save(os.path.join(
                sketch_path, sketch_name.split('/')[-1]))
    else:
        continue
