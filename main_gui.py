# ---------- IMPORT LIBRARIES ---------- #
import dlib
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import ttkbootstrap as ttk

# ---------- GLOBAL VARIABLES ---------- #
global capture              # video capture
global after_id             # id of the after function

# ---------- CUSTOM CLASSES ---------- #
class Camera():
    def __init__(self, width=640, height=480):
        """
        Creates a camera with the specified width and height
        Args:
            width (int, optional): width of the camera. Defaults to 640.
            height (int, optional): height of the camera. Defaults to 480.
        """
        self.width = width
        self.height = height
    
    def record(self) -> any:
        """
        Records the video from the camera
        Returns:
            any: the recorded video
        """
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return capture

    def size(self) -> tuple:
        """
        Returns the size of the camera
        Returns:
            tuple: (width, height)
        """
        return self.width, self.height

# ---------- CUSTOM FUNCTIONS ---------- #
def extract_index_nparray(nparray):
    """
    Extracts the index from a numpy array
    Args:
        nparray (numpy array): numpy array to extract the index from
    Returns:
        int: index of the numpy array
    """
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def default_camera(text=""):
    """
    Shows the default camera on the label
    Args:
        text (str, optional): text to show on the label. Defaults to "".
    """
    global after_id
    try:  
        app.after_cancel(after_id)                              # stop calling the function             
    except:
        pass
    text_widget.configure(text=text)                            # change the text of the label
    open_camera()                                               # show the camera on the label
    
def open_camera():
    """
    Shows the camera on the label
    """
    global capture
    global after_id
    if type(capture) == cv2.VideoCapture:
        _, frame = capture.read()                               # read the current frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)           # convert the frame to RGBA
        img = cv2.flip(img,1)                                   # flip the frame horizontally
        captured_img = Image.fromarray(img)                     # convert the frame to PIL format
        photo_image = ImageTk.PhotoImage(image=captured_img)    # convert the frame to Tkinter format
        camera_widget.photo_image = photo_image                 # keep a reference to the image to avoid garbage collection
        camera_widget.configure(image=photo_image)              # show the image on the label
        after_id = camera_widget.after(20, open_camera)         # call the function again after 20ms

def upload_image():
    """
    Upload an image from the file system to apply the face swap
    """
    img_path = askopenfilename(title="Select an image", filetypes=[
        ("image", ".jpg"),
        ("image", ".jpeg"),
        ("image", ".png")
    ])
    if img_path != "":
        img = cv2.imread(img_path)
        text = img_path + " used as reference"
        text_widget.configure(text=text)
        realtime_face_swap(img)
    
def realtime_face_swap(img):
    """
    Prepares the image to be swapped with the camera
    Args:
        img (_type_): image to swap the face with
    """
    global capture
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # the grayscale image has only one channel in 
                                                        # comparison with the color format so it's easier 
                                                        # to process for the CPU

    # 1) CREATE DETECTOR AND PREDICTOR
    # import detector to detect faces in the image (HOG-based)
    face_detector = dlib.get_frontal_face_detector()
    # import shape predictor to predict the location of 68 landmarks (points) on the face
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    # 2) FIND LANDMARKS IN THE REFERENCE IMAGE AND CREATE A CONVEX HULL
    try:
        # face_detector returns a list of rectangles (with top-left and right-bottom corners) that contain the bounding boxes of the faces in the image
        faces = face_detector(img_gray)
        # predict landmarks of the face using the shape predictor on the grayscale image
        landmark_points_ref = []
        for face in faces:
            landmarks = shape_predictor(img_gray, face)
            # for each landmark, get coordinates and store them in a list
            for p in range(0,68):
                x = landmarks.part(p).x
                y = landmarks.part(p).y
                # cv2.circle(img, (x,y), 3, (255,0,0), -1)                    # show the landmarks on the image
                landmark_points_ref.append((x,y))

        # convexhull -> external boundary of the points (the degrees inside cannot be bigger than 180°)
        # also called "minimum convex polygon"
        np_points_ref = np.array(landmark_points_ref,np.int32)              # convert the list of points to a numpy array
        convexhull_ref = cv2.convexHull(np_points_ref)                      # get the convex hull of the points
        # cv2.polylines(img, [convexhull], True, (255,0,0), 3)                # show the convex hull on the image


        # 3) FACE SEGMENTATION INTO TRIANGLES USING DELAUNAY TRIANGULATION
        rect = cv2.boundingRect(convexhull_ref)                             # get the bounding rectangle of the convex hull
        # (x,y,w,h) = rect                                                    # get the coordinates of the rectangle and draw it on the image
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        subdiv = cv2.Subdiv2D(rect)                                         # create a subdiv2D object with the rectangle
        subdiv.insert(landmark_points_ref)                                  # insert the landmark points
        triangles = subdiv.getTriangleList()                                # get the list of triangles
        triangles = np.array(triangles, dtype=np.int32)                     # convert the list to a numpy array

        # process each triangle
        triangles_indexes = []
        for t in triangles:
            pt1 = (t[0],t[1])
            pt2 = (t[2],t[3])
            pt3 = (t[4],t[5])

            # show the triangles on the image
            # cv2.line(img, pt1, pt2, (255,0,0), 1)
            # cv2.line(img, pt2, pt3, (255,0,0), 1)
            # cv2.line(img, pt3, pt1, (255,0,0), 1)

            # use coordinates to find index of the landmark points: where uses the value to find the index of the point in the array
            # the condition returns an array with the indexes of the points that satisfy the condition -> which point it might be
            # axis=1 returns the first element
            # the custom function returns only the value of the index
            index_pt1 = extract_index_nparray(np.where((np_points_ref == pt1).all(axis=1)))    
            index_pt2 = extract_index_nparray(np.where((np_points_ref == pt2).all(axis=1)))
            index_pt3 = extract_index_nparray(np.where((np_points_ref == pt3).all(axis=1)))

            # store the triangles in a list
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1,index_pt2,index_pt3]
                triangles_indexes.append(triangle)
        # 4) SWAPPING LOOP 
        app.after_cancel(after_id)                                  # stop calling the function
        swapping_loop(img, face_detector, shape_predictor, landmark_points_ref, triangles_indexes)
    except:
        default_camera("No face detected in the selected image")

def swapping_loop(img, face_detector, shape_predictor, landmark_points_ref, triangles_indexes):
    """
    Swaps the face in real time
    Args:
        img (_type_): image to swap the face with
        face_detector (_type_): face detector
        shape_predictor (_type_): shape predictor
        landmark_points_ref (_type_): landmark points of the reference image
        triangles_indexes (_type_): list of triangles with vertices indexes
    """
    global capture
    global after_id
    _, frame = capture.read()                               # read the current frame
    frame = cv2.flip(frame,1)                               # flip the frame horizontally
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     # convert the frame to grayscale
    new_face = np.zeros_like(frame)                         # create a new image with the same size as the frame

    # 5) GET LANDMARKS FROM THE CURRENT FRAME
    # process the frame to get the landmarks and convex hull
    faces = face_detector(gray_frame)
    landmark_points_frame = []
    for face in faces:
        landmarks = shape_predictor(gray_frame, face)
        for p in range(0,68):
            x = landmarks.part(p).x
            y = landmarks.part(p).y
            landmark_points_frame.append((x,y))

    # 6) APPLY TRIANGLES TO THE CURRENT FRAME IF THERE ARE LANDMARKS
    try:
        if (len(landmark_points_frame) != 0):
            np_points_frame = np.array(landmark_points_frame,np.int32)
            convexhull_frame = cv2.convexHull(np_points_frame) 
            # use triangles from the reference image and apply them to the current frame
            for indexes in triangles_indexes:
                # # SHOW TRIANGLES IN THE CURRENT FRAME
                # pt1 = landmark_points_frame[indexes[0]]
                # pt2 = landmark_points_frame[indexes[1]]
                # pt3 = landmark_points_frame[indexes[2]]

                # # # show the triangles on the frame
                # cv2.line(frame, pt1, pt2, (255,0,0), 1)
                # cv2.line(frame, pt2, pt3, (255,0,0), 1)
                # cv2.line(frame, pt3, pt1, (255,0,0), 1)
            
                #triangle in the reference image
                #get vertexes
                pt1_ref = landmark_points_ref[indexes[0]]
                pt2_ref = landmark_points_ref[indexes[1]]
                pt3_ref = landmark_points_ref[indexes[2]]
                # create an array with the triangle vertexes from the reference image
                triangle_ref = np.array([pt1_ref,pt2_ref,pt3_ref], np.int32)
                # get the bounding rectangle of the triangle
                rect_ref = cv2.boundingRect(triangle_ref)
                # get its coordinates
                (x_ref,y_ref,w_ref,h_ref) = rect_ref
                # crop triangle image
                cropped_triangle_ref = img[y_ref: y_ref + h_ref, x_ref: x_ref + w_ref]
                cropped_ref_mask = np.zeros((h_ref, w_ref), np.uint8)
                points_ref = np.array([[pt1_ref[0] - x_ref, pt1_ref[1] - y_ref],
                                        [pt2_ref[0] - x_ref, pt2_ref[1] - y_ref],
                                        [pt3_ref[0] - x_ref, pt3_ref[1] - y_ref]], np.int32)
                cv2.fillConvexPoly(cropped_ref_mask, points_ref, 255)
                cropped_ref = cv2.bitwise_and(cropped_triangle_ref, cropped_triangle_ref, mask=cropped_ref_mask)

                # triangle in the current frame
                # get vertexes
                pt1_frame = landmark_points_frame[indexes[0]]
                pt2_frame = landmark_points_frame[indexes[1]]
                pt3_frame = landmark_points_frame[indexes[2]]
                #create an array with the triangle vertexes from the current frame
                triangle_frame = np.array([pt1_frame,pt2_frame,pt3_frame], np.int32)
                # get the bounding rectangle of the triangle
                rect_frame = cv2.boundingRect(triangle_frame)
                # get its coordinates
                (x_frame,y_frame,w_frame,h_frame) = rect_frame
                # crop triangle image
                # cropped_triangle_frame = frame[y_frame: y_frame + h_frame, x_frame: x_frame + w_frame]
                cropped_frame_mask = np.zeros((h_frame, w_frame), np.uint8)
                points_frame = np.array([[pt1_frame[0] - x_frame, pt1_frame[1] - y_frame],
                                        [pt2_frame[0] - x_frame, pt2_frame[1] - y_frame],
                                        [pt3_frame[0] - x_frame, pt3_frame[1] - y_frame]], np.int32)
                cv2.fillConvexPoly(cropped_frame_mask, points_frame, 255)
                # cropped_frame = cv2.bitwise_and(cropped_triangle_frame, cropped_triangle_frame, mask=cropped_frame_mask)
                
                # warp triangle from the reference image to the current frame
                points_src = np.float32(points_ref)
                points_dst = np.float32(points_frame)
                M = cv2.getAffineTransform(points_src, points_dst)
                warped_triangle = cv2.warpAffine(cropped_ref, M, (w_frame, h_frame), flags=cv2.INTER_NEAREST)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_frame_mask)

                # apply the transformation to the frame
                area = new_face[y_frame: y_frame + h_frame, x_frame: x_frame + w_frame]             # get the area where the triangle will be applied
                area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)                                  # convert it to grayscale
                _, area_mask = cv2.threshold(area_gray, 1, 255, cv2.THRESH_BINARY_INV)              # create mask
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=area_mask) # apply the mask to the warped triangle
                triangle_area = cv2.add(area, warped_triangle)                                      # add the warped triangle to the area
                new_face[y_frame: y_frame + h_frame, x_frame: x_frame + w_frame] = triangle_area    # apply the area to the new face
      
            # make a mask of the face
            face_mask = np.zeros_like(gray_frame)                               # create a black image the same size of the frame
            head_mask = cv2.fillConvexPoly(face_mask, convexhull_frame, 255)    # fill the face with white
            face_mask = cv2.bitwise_not(head_mask)                              # invert the mask (face black, background white)

            # remove the face from the frame
            head_noface = cv2.bitwise_and(frame, frame, mask=face_mask)

            # add the new face to the frame
            result = cv2.add(head_noface, new_face)

            # get the center of the face and apply seamless cloning
            (x, y, w, h) = cv2.boundingRect(convexhull_frame)
            center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv2.seamlessClone(result, frame, head_mask, center_face, cv2.NORMAL_CLONE)
            frame = seamlessclone
    except:
        pass
        
    show_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)          # convert the frame to RGBA
    captured_img = Image.fromarray(show_img)                    # convert the frame to PIL format
    photo_image = ImageTk.PhotoImage(image=captured_img)        # convert the frame to Tkinter format
    camera_widget.photo_image = photo_image                     # keep a reference to the image to avoid garbage collection
    camera_widget.configure(image=photo_image)                  # show the image on the label
    after_id = camera_widget.after(20, lambda: swapping_loop(img, face_detector, shape_predictor, landmark_points_ref, triangles_indexes))  # call this function again in 20 milliseconds

        
# ---------- MAIN ------------ #

# CREATE THE CAMERA
cam = Camera()                                                      # create a camera object
capture = cam.record()                                              # record video from the camera

# DRAW THE GUI
app = ttk.Window(themename="minty", size=(800,800))                # create the GUI
app.title("Face Swapper - Camera")                                  # set the title 
app.bind('<Escape>', lambda e: app.quit())                          # press ESC to close the app
app.wm_iconphoto(True, ImageTk.PhotoImage(file="imgs/persona_speciale.png")) # set the icon (da cambiare o togliere dato che è meme)

camera_frame = ttk.Frame(app)                                       # create a frame
camera_frame.pack(pady=40)                                          # show the frame

label_camera_frame = ttk.LabelFrame(camera_frame, text="Camera")    # create a label frame
label_camera_frame.pack(pady=20, padx=20)                           # show the label frame

camera_widget = ttk.Label(label_camera_frame)                       # create a label to show the camera
camera_widget.pack(pady=10, padx=10)                                # show the label    

buttons_frame = ttk.Frame(app)                                      # create a frame
buttons_frame.pack()                                                # show the frame

upload_button = ttk.Button(buttons_frame, text="Upload image", width=20, command=upload_image, style="default") # create a button to upload an image
upload_button.pack()                                                # show the button

text_widget = ttk.Label(buttons_frame, text="")                     # create a label to show the text
text_widget.pack()                                                  # show the label

default_button = ttk.Button(buttons_frame, text="Default camera", width=20, command=default_camera, style="warning") # create a button to use the default camera
default_button.pack()                                               # show the button

# RUN THE APP
default_camera()                                                    # start the app with the default camera
app.mainloop()                                                      # run the app

'''
⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ 
⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ 
⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ 
⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠛⠉
'''

# TODO: - grafica - allineare il text_widget alla destra di upload_button
# TODO: - codice - permettere di fare il face swapping tra due foto senza dover uscire dall'app (menù principale???)
# TODO: - codice - migliorare la gestione dei punti della bocca (fede ha detto così)
# TODO: - codice - implementare il face swapping dati due volti rilevati nella camera