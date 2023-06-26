import cv2
import os

import cv2.data
from django.shortcuts import render
from django.http import HttpResponse
from .models import FaceRecognization
import face_recognition

def index(request):
    if request.method == "POST":
        if len(request.FILES) != 0:
            face = FaceRecognization()
            face.name = request.POST["username"]
            face.pan = request.FILES['pan_image']
            face.selfie = request.FILES['selfie_image']
            face.save()
            return HttpResponse("Details Added successfully")

        name = request.POST["username"]
        details = FaceRecognization.objects.get(name=name)

        # Set input and output directories
        input_directory = "F:/Images/PAN images/Input"
        output_directory = "F:/Images/PAN images/Output"

        os.makedirs(input_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        # Call the process_images function
        process_images(input_directory, output_directory)

        # Match faces and get result
        face_match_result = match_faces(details.pan.path, details.selfie.path)

        context = {
            "details": details,
            "face_match_result": face_match_result
        }

        return render(request, "index.html", context)
    else:
        return render(request, "index.html")


def process_images(input_directory, output_directory):
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Get the list of image files in the input directory
    image_files = os.listdir(input_directory)

    for filename in image_files:
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=0, minSize=(30, 30))
            counter = 0

            for (x, y, w, h) in faces:
                # Add some padding to the face region
                padding = 10
                x -= padding
                y -= padding
                w += padding * 3
                h += padding * 3

                # Ensure the cropped region is within the image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)

                # Crop the face region from the image
                face = image[y:y + h, x:x + w]

                # Display the cropped face
                cv2.imshow('Cropped Face', face)
                cv2.waitKey(2000)

                counter += 1
                print(f"The counter is {counter}")

                # Save the cropped face to the output directory
                output_path = os.path.join(output_directory, f"cropped_face_{counter}.jpg")
                cv2.imwrite(output_path, face)


                def match_faces(id_card_image, ref_image):
                    id_card = face_recognition.load_image_file(id_card_image)
                    ref = face_recognition.load_image_file(ref_image)

                    id_card_encodings = face_recognition.face_encodings(id_card)
                    ref_encodings = face_recognition.face_encodings(ref)

                    if len(id_card_encodings) == 0 or len(ref_encodings) == 0:
                        print("No faces found in the provided images.")
                        return

                    id_card_encoding = id_card_encodings[0]
                    ref_encoding = ref_encodings[0]

                    results = face_recognition.compare_faces([id_card_encoding], ref_encoding)
                    similarity = face_recognition.face_distance([id_card_encoding], ref_encoding)
                    similarity_percent = (1 - similarity[0]) * 100
                    
                    if results[0]:
                        return "Faces match with a similarity of {:.2f}%.".format(similarity_percent)
                    else:
                        return "Faces do not match."

# Assuming the following line is part of the `index` function
    face_match_result = match_faces(details.pan.path, details.selfie.path)
    context = {
    "details": details,
    "face_match_result": face_match_result
    }
    return render(request, "index.html", context)
