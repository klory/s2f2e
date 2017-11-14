from PIL import Image, ImageOps
photo = Image.open('./My FERET/00043.ppm')
print(photo.size)
photo_with_border = ImageOps.expand(photo, border=(100,0), fill='black')
print(photo_with_border.size)
photo_with_border.show()

width, height = photo.size

f = open('./CUFSF/photo_points/00043fb001d_931230.3pts')
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
left = max(0, points[0][0] - pupil_dist/1.5)
up = max(0, points[0][1] - eye_to_mouth_dist*1.5)
right = min(width, points[1][0] + pupil_dist/1.5)
down = min(height, points[2][1] + eye_to_mouth_dist/1.5)
image_cropped = photo_with_border.crop((left, up, right, down))
image_cropped = photo.crop((points[0][0], points[0][1], points[1][0], points[2][1]))
image_cropped.show()
