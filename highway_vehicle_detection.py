# !/usr/bin/env python
"""This script prompts a user to get highway vehicular data from a video"""

import os
import shutil
import cv2


def crossing_point(path, x_1, y_1, x_2, y_2):
    """Takes coordinates of line and returns the frame with the line drawn"""

    video = cv2.VideoCapture(path)
    cv2.namedWindow("Crossing Point Frame", cv2.WINDOW_NORMAL)

    for i in range(0, 1):
        success, image = video.read()
        width = image.shape[1]
        height = image.shape[0]
        print("Frame Resolution: " + str(width) + ":" + str(height))

        if x_1 != x_2 and y_1 != y_2:
            print("The line must be Vertical or Horizontal.")
            break

        if x_1 == x_2 and y_1 == y_2:
            print("The crossing line could not be a point.")
            break

        if x_1 < 0 or x_2 > width or y_1 < 0 or y_2 > height:
            print("Please enter the value within the video resolution limit of width "
                  + str(width) + " and height " + str(height) + ".")
            break

        cv2.line(image, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

        cv2.imshow("Crossing Point Frame", image)
        cv2.resizeWindow("Crossing Point Frame", (int(width / 2), int(height / 2)))

        key = cv2.waitKey(10000)
        # if q entered whole process will stop
        if key == ord('q'):
            break


# Function to extract frames
def threshold_frame(path, thresh_value, dil_value=5):
    """Takes path of the video, threshold value and dilation iteration and returns the gray frame,
     threshold frame and dilated frame"""
    # Capturing video
    video = cv2.VideoCapture(path)
    success = 1

    # Assigning our static_back to None
    static_back = None

    # Creating a resizable windows to display video
    cv2.namedWindow("Gray Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dilated Frame", cv2.WINDOW_NORMAL)

    # Infinite while loop to treat stack of image as video
    while success:
        # Reading frame(image) from video
        success, image = video.read()

        # Getting resolution of the video frames
        width = image.shape[1]
        height = image.shape[0]

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # In first iteration we assign the value of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue
        # Converting gray scale image to GaussianBlur so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Difference between static background and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and current frame is greater than 1,
        # it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, thresh_value, 255, cv2.THRESH_BINARY)[1]

        # We will dilate the image contours created by threshold frame with certain iteration
        dilate_frame = cv2.dilate(thresh_frame, None, iterations=dil_value)

        cv2.imshow("Gray Frame", gray)
        cv2.resizeWindow("Gray Frame", (int(width / 2), int(height / 2)))

        cv2.imshow("Threshold Frame", thresh_frame)
        cv2.resizeWindow("Threshold Frame", (int(width / 2), int(height / 2)))

        cv2.imshow("Dilated Frame", dilate_frame)
        cv2.resizeWindow("Dilated Frame", (int(width / 2), int(height / 2)))

        # We assign the value to of static back to gray again to compare it with next frame
        if static_back is not None:
            static_back = gray

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            break


def extract_images(path, thresh_value, x_1, y_1, x_2, y_2, dil_value=5, lower_limit=400):
    """Takes path of the video, threshold value, coordinates of line, dilation iteration
    and contour area lower limit and returns vehicle detection frame and saves the images"""
    # Capturing video
    video = cv2.VideoCapture(path)
    counter = 0
    success = 1

    # Assigning our static_back to None
    static_back = None

    # Creating a resizable windows to display video
    cv2.namedWindow("Extraction Frame", cv2.WINDOW_NORMAL)

    # Making a directory to save videos
    DIR = "Vehicle_Images"
    if os.path.exists(DIR):
        # This conditions ensures if there is no pre-existing directory and overwrite if exists
        shutil.rmtree(DIR)
    os.mkdir(DIR)

    # Change the directory that we created to save images
    os.chdir(DIR)

    # Infinite while loop to treat stack of image as video
    while success:
        # Reading frame(image) from video
        success, image = video.read()
        image1 = image.copy()
        # Getting resolution of the video frames
        width = image.shape[1]
        height = image.shape[0]

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur so that change can be find easily
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # In first iteration we assign the value of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Difference between static background and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and current frame is greater than thresh value,
        # it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, thresh_value, 255, cv2.THRESH_BINARY)[1]

        # We will dilate the image contours created by threshold frame with certain iteration.
        # Default iteration is set to 5.
        dilate_frame = cv2.dilate(thresh_frame, None, iterations=dil_value)

        cnts, _ = cv2.findContours(dilate_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # counting saving line
        cv2.line(image, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
        if x_1 != x_2 and y_1 != y_2:
            print("The line must be Vertical or Horizontal.")
            break
        if x_1 == x_2 and y_1 == y_2:
            print("The crossing line could not be a point.")
            break
        if x_1 < 0 or x_2 > width or y_1 < 0 or y_2 > height:
            print("Please enter the value within the video resolution limit of width "
                  + str(width) + " and height " + str(height) + ".")
            break

        # Running a for loop for finding contours in the frame
        for contour in cnts:
            # The default lower limit of contour area is 400.
            if cv2.contourArea(contour) < lower_limit:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)

            if x_1 == x_2 and y_1 != y_2:
                if x_1 <= x <= (x_1 + 1) or x_1 <= (x + w) <= (x_1 + 1):
                    if y > min(y_1, y_2) and (y + h) < max(y_1, y_2):
                        name = "vehicle_image_" + str(counter) + ".jpg"
                        crop_img = image1[y - 10:y + h + 10, x - 15:x + w + 15]
                        cv2.imwrite(name, crop_img)

                        counter += 1
                    # making green rectangle around the moving object
                cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

            elif y_1 == y_2 and x_1 != x_2:
                if y_1 <= y <= (y_1 + 1) or y_1 <= (y + h) <= (y_1 + 1):
                    if x > min(x_1, x_2) and (x + w) < max(x_1, x_2):
                        name = "vehicle_image_" + str(counter) + ".jpg"
                        crop_img = image1[y - 10:y + h + 10, x - 15:x + w + 15]
                        cv2.imwrite(name, crop_img)

                        counter += 1
                    # making green rectangle around the moving object
                cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

            else:
                continue

        cv2.putText(image, str(counter), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # We assign the value to of static back to gray again to compare it with next frame
        if static_back is not None:
            static_back = gray

        cv2.imshow("Extraction Frame", image)
        cv2.resizeWindow("Extraction Frame", (int(width / 2), int(height / 2)))

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            break


# Driver Code
if __name__ == '__main__':

    path = "--Put the video path here (relative/absolute)--"
    
    # Calling the function
    
    # crossing_point(path, x1, y1, x2, y2)
    # threshold_frame(path, thresh_value)
    # extract_images(path, thresh_value, x1, y1, x2, y2)

    
    # Press "q" to stop the vidoe frames
    
