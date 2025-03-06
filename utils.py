import os
from django.core.files.storage import FileSystemStorage
from accident_detection.settings import VIDEO_UPLOAD_PATH
import shutil

import cv2
import face_recognition
import pickle
from detection_app.block import block

face_path = "detection_app/accident_detection_model/trained_knn_model.xml"


def handle_uploaded_file(file,task):
    shutil.rmtree(VIDEO_UPLOAD_PATH)
    if not os.path.exists(VIDEO_UPLOAD_PATH):
        os.makedirs(VIDEO_UPLOAD_PATH)
    fs = FileSystemStorage(VIDEO_UPLOAD_PATH) #defaults to   MEDIA_ROOT  
    filename = fs.save(file.name, file)
    # print('saved')
    file_url = fs.url(filename)
    video_path = VIDEO_UPLOAD_PATH +'/'+ file.name
    if task == 'traffic':
        result = block(video_path)
    elif task == 'accident':
        result = detect_accident(video_path)

    return result


def mark_attendance(timePeriod):
    webcam = cv2.VideoCapture(0) 
    while True:
    # Loop until the camera is working
        rval = False
        while (not rval):
            # Put the image from the webcam into 'frame'
            (rval, frame) = webcam.read()
            if (not rval):
                print("Failed to open webcam. Trying again...")

        # Flip the image (optional)
        frame = cv2.flip(frame, 1)  # 0 = horizontal ,1 = vertical , -1 = both
        frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        predictions = predict(frame_copy, model_path="")  # add path here
        font = cv2.FONT_HERSHEY_DUPLEX
        for name, (top, right, bottom, left) in predictions:
            top *= 4  # scale back the frame since it was scaled to 1/4 in size
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, name, (left - 10, top - 6), font, 0.8, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    return name


def predict(img_path, knn_clf=None, model_path=None, threshold=0.5):  # 6 needs 40+ accuracy, 4 needs 60+ accuracy
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(face_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # Load image file and find face locations
    img = img_path
    face_box = face_recognition.face_locations(img)
    # If no faces are found in the image, return an empty result.
    if len(face_box) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
    
    # print(closest_distances)
    
    
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), face_box, matches
                )]


# import winsound
import cv2
import numpy as np

from .LoadModel import AccidentDetectionModel

json=r'.\detection_app\accident_detection_model\model.json'
model_weight=r'.\detection_app\accident_detection_model\model_weights.h5'
model = AccidentDetectionModel(json, model_weight)
font = cv2.FONT_HERSHEY_SIMPLEX
frequency = 2000
duration = 1000
output_directory = "detection_app/static/accident_frames"
static_path = "accident_frames"



def detect_accident(path = r".\video1.mp4"):
    video = cv2.VideoCapture(path) # for camera use video = cv2.VideoCapture(0)
    frame_count = 0
    frame_path = output_directory + f"/accident_frame_{frame_count}.jpg"
    static_img_path = static_path+ f"/accident_frame_{frame_count}.jpg"
    result = {}
    result= {
        'accident':False,
        'criminal_detected':False,
        'trafic_detected':False
    }
    while True:
        ret, frame = video.read()
        if not ret:
            print("frame reached end")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        predictions = predict(gray_frame, model_path="")
        for name, (top, right, bottom, left) in predictions:
            if(name!="unknown"):
                print(name)
                # result= {
                #     'criminal_detected':True,
                #     'criminal_name':name,
                #     'criminal_img_path':frame_path,
                #     'static_criminal_img':static_img_path,
                # }
                result['criminal_detected'] = True
                result['criminal_name'] = name
                result['criminal_img_path'] = frame_path
                result['static_criminal_img'] = static_img_path

                cv2.imwrite(frame_path, frame)
                print(f"Criminal frame saved: {frame_path}")
                return result 
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            if(98.5<= prob <= 99.6):
                # winsound.Beep(frequency, duration)
                
                print('accident detected')
                # Save the frame as an image
                
                cv2.imwrite(frame_path, frame)
                print(f"Accident frame saved: {frame_path}")
                frame_count += 1
                # result={
                #     'accident': True,
                #     'accident_img_path':frame_path,
                #     'static_accident_img':static_img_path,
                # }
                result['accident'] = True
                result['accident_img_path'] = frame_path
                result['static_accident_img'] = static_img_path
                return result

            # cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            # cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)
    return result
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return
        # cv2.imshow('Video', frame)  





import os
import io, base64
from PIL import Image
import re
def upload_img(images,imgPath):
    pattern = '^data:image/[^;]+;base64,'
    for i ,img in enumerate(images): 
        print(i, str(i))
        base64_str =  re.sub(pattern, '', img)
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        img.save(imgPath+'/'+str(i)+'.png')




import math
from sklearn import neighbors
import os
import os.path
import pickle

from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory

    train_dir = os.listdir('images/')
    print(train_dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("images/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            print("images/" + person + "/" + person_img)
            face = face_recognition.load_image_file("images/" + person + "/" + person_img)
            print(face.shape)
            # Assume the whole image is the location of the face
            # height, width, _ = face.shape
            # location is in css order - top, right, bottom, left
            height, width, _ = face.shape
            face_location = (0, width, height, 0)
            print(width, height)

            face_enc = face_recognition.face_encodings(face)

            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()

            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

    print(np.array(encodings).shape)
    print(np.array(names).shape)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def face_train():
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("images", model_save_path="detection_app/accident_detection_model/trained_knn_model.xml", n_neighbors=2)
    print("Training complete!")




