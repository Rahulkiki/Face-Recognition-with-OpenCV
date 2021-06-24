import numpy as np
import cv2 as cv
import os
#Euclidean Distance Function
def dist(x,y):
    return np.sqrt(sum((y-x)**2))

# K-th Nearest Neighbours Function
def knn( x , y , query , k = 5 ):
    val = []
    size = x.shape[0]

    #Iterate in Training Set to store distance from reference Point to all other Points
    for i in range(size):
        d = dist(query,x[i])
        val.append([d,y[i]])

    #Sorting list for getting K-th Values
    val = sorted(val)[:k]
    #Converting list to Numpy Array 
    val = np.array(val)

    # Frequency Array for counting Values
    new_vals = np.unique( val[:,1] , return_counts = True )

    #Getting index of the value containing most values
    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred

#For Capturing Video
video = cv.VideoCapture(0)
#For Object Classifier
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

label = []
#For mapping of names with class_id
names = {}
#For storing faces 
face_data = []
class_id = 0

for file in os.listdir("./"):
    if file.endswith(".npy") :
        # print("loaded "+ file[:-4])
        names[ class_id ] = file[ :-4 ]
        data_item = np.load( file )
        face_data.append( data_item )

        target = class_id*np.ones(( data_item.shape[0] ),)
        class_id += 1
        label.append( target )

face_dataset = np.concatenate( face_data, axis = 0 )
label_dataset = np.concatenate( label , axis = 0 ).reshape((-1,1))

#TESTING PART

while True:
    ret , frame = video.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale( frame , 1.3 , 5 )

    for face in faces:
        x , y , w , h = face
        
        offset = 10
        face_size = frame[ y-offset : y+h+offset , x-offset : x+w+offset ]
        face_size = cv.resize( face_size , (100,100) )

        print( face_size.flatten() )
        out = knn( face_dataset , label_dataset , face_size.flatten() )

        pred_name = names[int(out)]
        cv.putText( frame , pred_name , (x,y-10) , cv.FONT_HERSHEY_SIMPLEX , 2 , (255,0,0) , cv.LINE_AA )
        cv.rectangle( frame , (x,y) , (x+w,y+h) , (255,255,0) , 2 )

    cv.imshow("Faces",frame)

    key_press = cv.waitKey(1) & 0xFF
    if key_press == ord('q'):
        break

video.release()
cv.destroyAllWindows()
