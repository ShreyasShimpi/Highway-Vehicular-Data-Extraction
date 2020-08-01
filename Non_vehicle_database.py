# Python program to implement Highway Vehicle Detection

import os
import shutil
import cv2
import random


counter = 3001
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Capturing video
Videopath = input()
video = cv2.VideoCapture(Videopath)

frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(frames)

# Make another directory for saving vehicle images
dir = "Non_Vehicle_images"
if not os.path.exists(dir):
    os.mkdir(dir)
os.chdir(dir)

# Infinite while loop to treat stack of image as video
while True:
    # Reading frame(image) from video
    check, frame = video.read()

    x = random.randint(0, 1800)
    y = random.randint(400, 900)
    w = random.randint(50, 120)
    h = random.randint(60, 90)

    # cv2.rectangle(frame, (0, 600), (1980, 850), (255, 255, 255), 2)

    name = "non_vehicle_image_" + str(counter) + ".jpg"
    crop_img = frame[y:y + h, x:x + w]
    cv2.imwrite(name, crop_img)

    # print(x, y, h, w)
    # print(counter)
    if counter >= 4000:
        print(counter)
        break
    counter += 1

    # making green rectangle around the moving object
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
