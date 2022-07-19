

from ast import excepthandler
from pickle import TRUE
from matplotlib import pyplot as plt
import cv2
import numpy as np
import imutils
import easyocr
from csv import writer
import os.path

i = 1

while(TRUE):
#run code
  while not os.path.isfile("C:\\Users\\matth\\OneDrive\\Documents\\GitHub\\ITC303-Team-07\\Test images\\image"+str(i)+".jpg"):
    #ignore if no such file is present.
    break
  img = cv2.imread("C:\\Users\\matth\\OneDrive\\Documents\\GitHub\\ITC303-Team-07\\Test images\\image"+str(i)+".jpg")
  i += 1
  # %% [markdown]
  # **1. Read in Image, Grayscale and Blur**

  # %%
  #img = cv2.imread('/content/ITC303-Team-07/Test images/' + 'image' + image + '.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

  # %% [markdown]
  # **2. Apply filter and find edges for localization**

  # %%
  bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
  edged = cv2.Canny(bfilter, 30, 200) #Edge detection
  plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

  # %% [markdown]
  # **3. Find Contours and Apply Mask**

  # %%
  keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

  # %%
  location = None
  for contour in contours:
      approx = cv2.approxPolyDP(contour, 10, True)
      if len(approx) == 4:
        location = approx
        break

  # %%
  location

  # %%
  mask = np.zeros(gray.shape, np.uint8)
  new_image = cv2.drawContours(mask, [location], 0, 255, -1)
  new_image = cv2.bitwise_and(img, img, mask=mask)

  # %%
  plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

  # %%
  (x,y) = np.where(mask==255)
  (x1, y1) = (np.min(x), np.min(y))
  (x2, y2) = (np.max(x), np.max(y))
  cropped_image = gray[x1:x2+1, y1:y2+1]

  # %%
  plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

  # %% [markdown]
  # **4. Use Easy OCR to Read Text**

  # %%
  reader = easyocr.Reader(['en'])
  result = reader.readtext(cropped_image)
  print(result[0][-2])

  # %% [markdown]
  # **5. Render Result**

  # %%
  text = result[0][-2]
  font = cv2.FONT_HERSHEY_SIMPLEX
  res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
  res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
  plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


  
  # Open our existing CSV file in append mode
  # Create a file object for this file
  with open('result.csv', 'a') as f_object:
  
    np.savetxt(f_object, result, delimiter=",", fmt='% s')
  
    #Close the file object
    f_object.close()
