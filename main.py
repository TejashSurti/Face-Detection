import cv2

image = cv2.imread('road.jpg')

h, w = image.shape[:2]

print("Height = {}, Width{}".format(h,w))

# Extract rgb value

(B, G, R) = image[100,100]

print("R = {}, G = {}, B = {}".format(R,G,G))

# Region of interest

roi = image[100:500, 200:700]

# resize the image

resize = cv2.resize(image, (800,800))

# rotating the image

center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, -45,1.0)

#performing affline transforation
rotated = cv2.warpAffine(image,matrix, (w,h))

# drawing a rectangle

output = image.copy()

rectangle = cv2.rectangle(output, (1500,900), (600,400), (255,0,0), 2)

#Displaying teXT

output = image.copy()

text = cv2.putText(output, 'OpenCV Demo', (500,500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 2)


