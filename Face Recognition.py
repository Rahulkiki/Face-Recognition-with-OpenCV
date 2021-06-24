import cv2 as cv
import numpy as np

#Capture Video 
video = cv.VideoCapture(0)
#Trained Face detectorRahul
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
skip = 0
file_name = input("Enter the name : ")

while True:
    #return status and Frame of video
    ret , frame = video.read()

    #if couldn't capture properly return False
    if ret == False:
        continue

    #Detect Object with Haar-Cascade-Classifier 
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    #sorting the Multiple frames according to their size  (W*H) 
    sorted( faces ,key = lambda f:f[2]*f[3] )
    
    for face in faces[-1:]:
        x,y,w,h = face
        # 1. Drawing Rectangle on captured Faces 
        cv.rectangle(faces,(x,y),(x+w,y+h),(0,255,255),2)

        # 2. Cutting Out the Required Size window for Saving Face information
        offset = 10
        # 3. Slicing from frame with more window size
        face_size = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        # 4. Resizing for new window for Cutout Face
        face_size = cv.resize(face_size,(100,100))

        if skip%10 == 0:
            face_data.append(face_size)
            print(len(face_data))

    #Displaying Camera and small Camera for required size face
    cv.imshow("Frame",frame)
    cv.imshow("Face Cutout",face_data[-1])
    
    #for stoping Camera
    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#converted array into Numpy array for saving and facilities
face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0],-1)

#File Saving
np.save(file_name + ".npy" ,face_data)
print("Successful")

#Releasing Captured Video
video.release()
cv.destroyAllWindows()