from utility import*

def get_face_n_eyes(photo):
    image = np.copy(photo)
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/anto/University/Driving-Visual-Attention/data/data")  

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
    pitch, yaw, roll = model.predict(image)
    if doPlot:
        model.draw_axis(image, yaw, pitch, roll)
        return image
    else:
        return pitch[0], yaw[0], roll[0]
    

def get_eyes(photo, predictor, face_detector):
    # Detect faces in the image
    faces = face_detector(photo, 1)
    # Ensure only one face is detected
    if len(faces) == 0:
        print("No face detected")
        return None
    elif len(faces) > 1:
        #face = faces[0] #prendi la prima faccia
        print("More than one face detected")
        return None
    else:
        face = faces[0] #prendi l'unica faccia 
        # Get facial landmarks for the detected face
        landmarks = predictor(photo, faces[0])
        # Get the face bounding box from the detector
        x1 = face.left()    # Punto più a sinistra
        y1 = face.top()     # Punto più in alto
        x2 = face.right()   # Punto più a destra
        y2 = face.bottom()  # Punto più in basso
        # Estrai e salva le immagini degli occhi
        for i in range(2):  # Due occhi
            # Coordinate dell'occhio
            x1 = landmarks.part(36 + i * 6).x
            y1 = landmarks.part(37 + i * 6).y
            x2 = landmarks.part(39 + i * 6).x
            y2 = landmarks.part(40 + i * 6).y
            if i==0:
                left_eye = photo[y1-8:y2+5, x1:x2]
                # x y della pupilla sono stati presi as midpoint between landmarks corresponding to the corners of the eyes
                pupil_left = (
                    landmarks.part(36 + i * 6).x + landmarks.part(39 + i * 6).x) // 2, (landmarks.part(37 + i * 6).y + landmarks.part(40 + i * 6).y) // 2
            if i==1:
                pupil_right = (
                    landmarks.part(42 + i * 6).x + landmarks.part(45 + i * 6).x) // 2, (landmarks.part(43 + i * 6).y + landmarks.part(46 + i * 6).y) // 2
        return [left_eye, pupil_left, pupil_right]


