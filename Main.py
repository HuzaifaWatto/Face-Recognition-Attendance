from datetime import datetime
import face_recognition
import  numpy as  np
import cv2
import csv

video_capture = cv2.VideoCapture(0)


# Load Known Face
image1 = face_recognition.load_image_file("faces/1.png")
image1_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file("faces/2.png")
image2_encoding = face_recognition.face_encodings(image2)[0]

known_faces_encodings = [image1_encoding, image2_encoding]
known_faces_names = ["1","2"]

# list of expe stu
students = known_faces_names.copy()

face_locations = []
face_encodings = []

# get current datetime
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx = .25, fy =.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize face
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index= np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # add text if presant
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10,100)
            fscale = 1.5
            fcolor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name + "Presant", bottom_left_corner, font, fscale, fcolor, thickness, linetype)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
