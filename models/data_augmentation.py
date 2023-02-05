import cv2
import glob


def extract_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # select the path
    path = "./../../img_align_celeba/*.*"
    img_number = 1  # Start an iterator for image number.
    img_list = glob.glob(path)
    # Extract faces from a subset of images to be used for training.
    for file in img_list:
        img = cv2.imread(file, 1)  # now, we can read each file since we have the full path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_color = img[y:y + h, x:x + w]
            resized = cv2.resize(roi_color, (64, 64))
            images = []
            cv2.imwrite("./../../extracted_faces/" + str(img_number) + ".jpg", resized)
            images.append(resized)
        img_number += 1


if __name__ == "__main__":
    extract_faces()
