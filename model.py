import os
import pickle
import sys
import face_recognition

def recording(name):

    if not os.path.exists("dataset"):
        print("[Error] not found directory")
        sys.exit()

    encodings = []
    images = os.listdir("dataset")

    for (i, image) in enumerate(images):
        print(f"[+] {i+1}/{len(images)}")

        face_img = face_recognition.load_image_file(f"dataset/{image}")
        face_enc = face_recognition.face_encodings(face_img)[0]

        if len(encodings) == 0:
            encodings.append(face_enc)
        else:
            for j in range(0, len(encodings)):
                res = face_recognition.compare_faces([face_enc], encodings[j])

                if res[0]:
                    encodings.append(face_enc)
                    print("It's u!")
                    break
                else:
                    print("Who r u?")
                    break

    data = {
        "name": name,
        "encodings": encodings
    }

    with open(f"{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"File {name}_encodings.pickle created"


def main():
    print(recording("Robert Downey Jr"))

if __name__ == '__main__':
    main()