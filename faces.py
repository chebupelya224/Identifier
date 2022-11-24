import face_recognition
from PIL import Image, ImageDraw
import os
import pickle
import cv2


def face_rec(img_path):

    face_img = face_recognition.load_image_file(img_path)
    face_img_location = face_recognition.face_locations(face_img)

    print(f"Found {len(face_img_location)} face(s) in this image")

    pil_img1 = Image.fromarray(face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for(top,right, bottom, left) in face_img_location:
        draw1.rectangle(((left, top), (right, bottom)), outline=(0,102,204), width=4)

    del draw1
    pil_img1.save("found_image/new_face.jpeg")


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]
    print(img2_encodings)

    res = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(res)


def face_capture():
    cascade_path = 'haarcascade_frontalface_default.xml'

    count = 0

    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    clf = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture('Robs.mp4')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fps = cap.get(cv2.CAP_PROP_FPS)
        scr = fps * 3

        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if ret:

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            frame_id = int(round(cap.get(1)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(20)

            if frame_id % scr == 0:
                cv2.imwrite(f"dataset/{count}screen.jpeg", frame)
                print(f"screenshot {count} was taken")
                count += 1

            if key == ord(" "):
                cv2.imwrite(f"dataset/{count}extrasrceen.jpeg", frame)
                print(f"extrascreenshot {count} was taken")
                count += 1

            if key == ord("q"):
                break

        else:
            print("[Error]")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    face_capture()

if __name__ == '__main__':
    main()