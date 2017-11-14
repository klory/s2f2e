import glob
import os
import re

photo_link = "./CUFSF/names_of_photos_used_for_drawing_sketches.txt"
f = open(photo_link, 'r')
photo_names = f.readlines()
photo_names = map(str.strip, photo_names)
f.close()

for photo_name in photo_names:
	m = re.findall('\d+', photo_name)
	photo_idx = m[0]
	photo_date = m[-1]
	m = re.findall('[a-z]+', photo_name)
	photo_pose = m[0]
	path = ''.join(['./FERET/colorferet/colorferet/dvd1/data/images/', photo_idx, '/*.ppm.bz2'])
	names = glob.glob(path)
	matches = [s for s in names if photo_date in s]
	matches = [s for s in matches if photo_pose in s]
	if len(matches) == 0:
		continue
	os.rename(matches[0], os.path.join('./My FERET', matches[0].split('/')[-1]))
