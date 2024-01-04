from utility import*

def get_face_n_eyes(photo, face_detector, predictor):
    image = np.copy(photo)  
    # Detect faces in the image
    faces = face_detector(image, 1)
    face = faces[0]
    # Get facial landmarks for the detected face
    landmarks = predictor(image, face)
    # Get the face bounding box from the detector
    x1 = face.left()    # Punto più a sinistra
    y1 = face.top()     # Punto più in alto
    x2 = face.right()   # Punto più a destra
    y2 = face.bottom()  # Punto più in basso
    face_image = image[y1:y2, x1:x2]
    # Estrai e salva le immagini degli occhi
    for i in range(2):  # Due occhi
        # Coordinate dell'occhio
        x1 = landmarks.part(36 + i * 6).x
        y1 = landmarks.part(37 + i * 6).y
        x2 = landmarks.part(39 + i * 6).x
        y2 = landmarks.part(40 + i * 6).y
        if i==0:
            left_eye = image[y1-8:y2+5, x1:x2]
            # x y della pupilla sono stati presi as midpoint between landmarks corresponding to the corners of the eyes
            pupil_left = (
                landmarks.part(36 + i * 6).x + landmarks.part(39 + i * 6).x) // 2, (landmarks.part(37 + i * 6).y + landmarks.part(40 + i * 6).y) // 2
        if i==1:
            right_eye = image[y1-8:y2+5, x1:x2]
            pupil_right = (
                landmarks.part(42 + i * 6).x + landmarks.part(45 + i * 6).x) // 2, (landmarks.part(43 + i * 6).y + landmarks.part(46 + i * 6).y) // 2

    landmarks = [(landmark.x, landmark.y) for landmark in landmarks.parts()]
    return face_image, left_eye, right_eye, pupil_left, pupil_right, landmarks
    
    
def get_headpose(image, model, doPlot = False):
    if doPlot:
        pitch, yaw, roll = model.predict(image)
        model.draw_axis(image, yaw, pitch, roll)
        return image
    else:
        rgb = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        x, y, w, h = 25, 100, 775, 900
        cropped = rgb[y:y+h, x:x+w]      
        pitch, yaw, roll = model.predict(cropped)
        return pitch[0], yaw[0], roll[0]
    

def get_eyes(image, predictor, face_detector):
    original_image = np.copy(image)
    x, y, w, h = 25, 100, 775, 900
    cropped = image[y:y+h, x:x+w] # Resize image and focus only on the region where the face is 
    gray = cv2.cvtColor(src=cropped, code=cv2.COLOR_BGR2GRAY)     
    faces = face_detector(gray)
    # Check the number of faces detected
    if len(faces) != 1:
        print("No face detected")
        # If zero or more than one face detected, return None
        return None
    else:
        face = faces[0]
        landmarks = predictor(gray, face)
        # Extract the left eye region directly from the original image
        left_eye = original_image[y + landmarks.part(37).y - 8:y + landmarks.part(40).y + 5,
                          x + landmarks.part(36).x:x + landmarks.part(39).x]
        # Calculate pupil coordinates for the left eye
        pupil_left = (
            x + landmarks.part(36).x + landmarks.part(39).x // 2,
            y + landmarks.part(37).y + landmarks.part(40).y // 2
        )
        # Calculate pupil coordinates for the right eye
        pupil_right = (
            x + landmarks.part(42).x + landmarks.part(45).x // 2,
            y + landmarks.part(43).y + landmarks.part(46).y // 2
        )
        return [left_eye, pupil_left, pupil_right]
    
    


