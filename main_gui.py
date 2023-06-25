# ---------- IMPORT LIBRARIES ---------- #
import dlib
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap import DoubleVar
 
# ---------- GLOBAL VARIABLES ---------- #
global capture              # video capture
global stored_frame        # current frame to be shown on the label
global after_id             # id of the function called after a delay
global face_detector        # detector to detect faces in the image (HOG-based)
global shape_predictor      # predictor to predict the location of 68 landmarks (points) on the face
global scale_value          # value of the scale
global swap_active          # True if the face swap is active
global cartoon_active       # True if the cartoon filter is active
global eye_active           # True if the eye filter is active
global splash_active        # True if the splash filter is active
global img_path             # path of the image for the face swapping
                            # global because it's needed to regenerate the face swap after changing filter
global milsec               # milliseconds between each frame

# ---------- CUSTOM CLASSES ---------- #
class Camera():
    def __init__(self, width:int=640, height:int=480):
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
def default_camera(text:str="") -> None:
    """
    Shows the default camera on the label
    Args:
        text (str, optional): text to show on the label. Defaults to "".
    """
    global after_id
    global swap_active
    global cartoon_active
    global img_path
    
    img_path = ""                               # reset the image path
    swap_active = False                         # reset the face swap filter
    if cartoon_active:                          # if the cartoon filter is active
        scale.pack_forget()                     # hide the scale
        scale_title.pack_forget()               # hide the scale title
        s_value.pack_forget()                   # hide the scale value
    remove_filters()                            # remove all the filters
    try:  
        app.after_cancel(after_id)              # stop calling the function             
    except Exception as e:
        # print("Default camera ",e)
        pass
    text_widget.configure(text=text)            # change the text of the label
    open_camera()                               # show the camera on the label
    
def open_camera() -> None:
    """
    Opens the camera and shows the current frame on the label
    """
    global capture
    global after_id
    global stored_frame
    if type(capture) == cv2.VideoCapture:
        _, frame = capture.read()                               # read the current frame
        img = cv2.flip(frame,1)                                 # flip the frame horizontally
        stored_frame = img                                     # store the current frame
        put_frame()                                             # show the frame
        after_id = camera_widget.after(milsec, open_camera)     # call the function again

def upload_image() -> None:
    """
    Upload an image from the file system to apply the face swap
    """
    global img_path
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

def put_frame() -> None:
    """
    Puts the current frame on the label
    """
    global stored_frame
    show_img = cv2.cvtColor(stored_frame, cv2.COLOR_BGR2RGBA)  # convert the frame to RGBA
    captured_img = Image.fromarray(show_img)                    # convert the frame to PIL format
    photo_image = ImageTk.PhotoImage(image=captured_img)        # convert the frame to Tkinter format
    camera_widget.photo_image = photo_image                     # keep a reference to the image to avoid garbage collection
    camera_widget.configure(image=photo_image)                  # show the image on the label
    
def detect_facial_landmarks(img_gray:np.ndarray) -> list:
    """
    Detects the facial landmarks on the image
    Args:
        img_gray (ndarray): grayscale image
    Returns:
        list: list of the facial landmarks
    """
    # use the biggest face detected
    face = max(face_detector(img_gray), key=lambda r: r.area())
    # predict landmarks of the face using the shape predictor on the grayscale image
    landmarks = shape_predictor(img_gray, face)
    # for each one of the 68 landmarks, get coordinates and store them in a list
    landmark_points_ref = []
    for p in range(0,68):
        x = landmarks.part(p).x
        y = landmarks.part(p).y
        # cv2.circle(img, (x,y), 3, (255,0,0), -1)                        # show the landmarks on the image
        landmark_points_ref.append((x,y))
    return landmark_points_ref

def get_cropped_triangle(img:np.ndarray, landmarks:list, v1:int, v2:int, v3:int) -> tuple:
    """
    Gets the cropped triangle and other info from the image
    Args:
        img (ndarray): image
        landmarks (list): list of the facial landmarks
        v1 (int): vertex 1
        v2 (int): vertex 2
        v3 (int): vertex 3
    Returns:
        tuple: (points, cropped, cropped_mask, x, y, w, h)
    """
    # get coordinates
    pt1 = landmarks[v1]
    pt2 = landmarks[v2]
    pt3 = landmarks[v3]
    # create an array with the triangle vertexes from the image
    triangle = np.array([pt1,pt2,pt3], np.int32)
    # get the bounding rectangle of the triangle
    rect = cv2.boundingRect(triangle)
    # get its size
    (x,y,w,h) = rect
    # get the part of the image with the triangle
    cropped_triangle = img[y: y + h, x: x + w]
    # create a mask with the triangle
    cropped_mask = np.zeros((h, w), np.uint8)
    # get the points of the triangle
    points = np.array([[pt1[0] - x, pt1[1] - y],
                        [pt2[0] - x, pt2[1] - y],
                        [pt3[0] - x, pt3[1] - y]], np.int32)
    # fill the triangle with white
    cv2.fillConvexPoly(cropped_mask, points, 255)
    # apply the mask to the triangle
    cropped = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_mask)
    return points, cropped, cropped_mask, x, y, w, h

def realtime_face_swap(img:np.ndarray) -> None:
    """
    Prepares the image to be swapped with the camera
    Args:
        img (ndarray): image
    """
    global after_id
    global face_detector
    global shape_predictor
    global swap_active

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
        # find landmarks of the reference face
        landmark_points_ref = detect_facial_landmarks(img_gray)

        # convexhull -> external boundary of the points (the degrees inside cannot be bigger than 180°)
        # also called "minimum convex polygon"
        np_points_ref = np.array(landmark_points_ref,np.int32)              # convert the list of points to a numpy array
        convexhull_ref = cv2.convexHull(np_points_ref)                      # create the convex hull
        # cv2.polylines(img, [convexhull], True, (255,0,0), 3)                # show the convex hull on the image

    # 3) FACE SEGMENTATION INTO TRIANGLES USING DELAUNAY TRIANGULATION
        rect = cv2.boundingRect(convexhull_ref)                             # get the bounding rectangle of the convex hull
        # (x,y,w,h) = rect                                                    # get the coordinates of the rectangle and draw it on the image
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        subdiv = cv2.Subdiv2D(rect)                                         # create a subdiv2D object with the rectangle
        subdiv.insert(landmark_points_ref)                                  # insert the landmark points
        triangles = subdiv.getTriangleList()                                # get the list of Delaunay triangles
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
            index_pt1 = np.where((np_points_ref == pt1).all(axis=1))[0][0]
            index_pt2 = np.where((np_points_ref == pt2).all(axis=1))[0][0]
            index_pt3 = np.where((np_points_ref == pt3).all(axis=1))[0][0]

            # store the triangles in a list
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1,index_pt2,index_pt3]
                triangles_indexes.append(triangle)

        # remove duplicates
        triangles_indexes = list(set(tuple(t) for t in triangles_indexes))

    # 4) SWAPPING LOOP 
        app.after_cancel(after_id)                                  # stop calling the function
        swap_active = True                                          # set the flag to True
        swapping_loop(img, landmark_points_ref, triangles_indexes)  # call the swapping loop
    except Exception as e:
        # print("Reference image exception: ",e)
        default_camera("No face detected in the selected image")

def swapping_loop(img:np.ndarray, landmark_points_ref:list, triangles_indexes:list) -> None:
    """
    Swaps the face in real time
    Args:
        img (ndarray): image to swap the face with
        landmark_points_ref (list): landmark points of the reference image
        triangles_indexes (list): list of triangles with vertices indexes
    """
    global capture
    global after_id
    global stored_frame
    global cartoon_active
    global eye_active

    _, frame = capture.read()                               # read the current frame
    frame = cv2.flip(frame,1)                               # flip the frame horizontally
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     # convert the frame to grayscale
    new_face = np.zeros_like(frame)                         # create a new image with the same size as the frame
    stored_frame = frame                                   # store the current frame

    # points of the mouth
    mouth_points = [
    # [48],  # <outer mouth>
    # [49],
    # [50],
    # [51],
    # [52],
    # [53],
    # [54],
    # [55],
    # [56],
    # [57],
    # [58],  # </outer mouth>
    [59],  # <inner mouth>
    [60],
    [61],
    [62],
    [63],
    [64],
    [65],
    [66],
    [67],  # </inner mouth>
]
    # points of the eyes
    eyes_points = [
    [36],  # <left eye>
    [37],
    [38],
    [39],
    [40],
    [41],  # </left eye>
    [42],  # <right eye>
    [43],
    [44],
    [45],
    [46],
    [47],  # </right eye>
]
    mouth_points_set = set(mp[0] for mp in mouth_points)    # convert the list of mouth points to a set
    eyes_points_set = set(ep[0] for ep in eyes_points)      # convert the list of eyes points to a set

    try:
    # 5) GET LANDMARKS FROM THE CURRENT FRAME
        # process the frame to get the landmarks
        landmark_points_frame = detect_facial_landmarks(gray_frame)

    # 6) APPLY TRIANGLES TO THE CURRENT FRAME IF FACE IS DETECTED
        if (len(landmark_points_frame) != 0):
            # create a convex hull of the landmarks in the current frame
            np_points_frame = np.array(landmark_points_frame,np.int32)
            convexhull_frame = cv2.convexHull(np_points_frame) 
            # use triangles from the reference image and apply them to the current frame
            for indexes in triangles_indexes:
                v1 = indexes[0]
                v2 = indexes[1]
                v3 = indexes[2]

                # SHOW TRIANGLES IN THE CURRENT FRAME
                # pt1 = landmark_points_frame[v1]
                # pt2 = landmark_points_frame[v2]
                # pt3 = landmark_points_frame[v3]
                # cv2.line(frame, pt1, pt2, (255,0,0), 1)
                # cv2.line(frame, pt2, pt3, (255,0,0), 1)
                # cv2.line(frame, pt3, pt1, (255,0,0), 1)
            
                # triangle in the reference image
                points_ref, cropped_ref, _, _, _, _, _ = get_cropped_triangle(img, landmark_points_ref, v1, v2, v3)
                
                # triangle in the current frame
                points_frame, cropped_frame, cropped_frame_mask, x_frame, y_frame, w_frame, h_frame = get_cropped_triangle(frame, landmark_points_frame, v1, v2, v3)
                
                # check if the triangle is in the mouth area - no need to warp it
                if (v1 in mouth_points_set and v2 in mouth_points_set and v3 in mouth_points_set) or (v1 in eyes_points_set and v2 in eyes_points_set and v3 in eyes_points_set):
                    warped_triangle = cropped_frame
                # else warp triangle from the reference image to the current frame
                else:
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
            convexhull_mouth = cv2.convexHull(np_points_frame[60:])             # get the convex hull of the mouth
            convexhull_left_eye = cv2.convexHull(np_points_frame[36:42])        # get the convex hull of the left eye
            convexhull_right_eye = cv2.convexHull(np_points_frame[42:48])       # get the convex hull of the right eye
            head_mask = cv2.fillConvexPoly(head_mask, convexhull_mouth, 0)      # fill the mouth with black
            head_mask = cv2.fillConvexPoly(head_mask, convexhull_left_eye, 0)   # fill the left eye with black
            head_mask = cv2.fillConvexPoly(head_mask, convexhull_right_eye, 0)  # fill the right eye with black
            face_mask = cv2.bitwise_not(head_mask)                              # invert the mask (face and mouth black, background white)

            # remove the face from the frame
            head_noface = cv2.bitwise_and(frame, frame, mask=face_mask)

            # add the new face to the frame
            result = cv2.add(head_noface, new_face)

            # get the center of the face and apply seamless cloning
            (x, y, w, h) = cv2.boundingRect(convexhull_frame)
            center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv2.seamlessClone(result, frame, head_mask, center_face, cv2.NORMAL_CLONE)
            stored_frame = seamlessclone

    except Exception as e:
        # print("Frame exception: ",e)
        pass

    if cartoon_active:                          # if cartoonize is active
        cartoonize_frame()                      # cartoonize the frame

    elif splash_active:                           # if splash is active
        splash()                                  # splash the frame

    elif eye_active:                            # if eye swap is active
        change_eyes()                           # change the eyes
    put_frame()                                 # show the frame
    after_id = camera_widget.after(milsec, lambda: swapping_loop(img, landmark_points_ref, triangles_indexes))  # call this function again

def cartoonize_frame() -> None:
    """
    Cartoonize the current frame
    """
    global stored_frame
    global after_id
    global cartoon_active
    global swap_active
    global scale_value

    remove_filters()                                        # remove all the filters
    cartoon_active = True                                   # set cartoonize as active
    if not swap_active:                                     # if swap is not active
        app.after_cancel(after_id)                              # stop calling the function
        _, fr = capture.read()                                  # read the current frame
        fr = cv2.flip(fr,1)                                     # flip the frame horizontally
    else:                                                   # if swap is active
        fr = stored_frame                                       # use the stored frame
    pack_scale(1,10,"Blur")                                 # pack the scale widget
    gray = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)              # convert the frame to grayscale
    gray = cv2.medianBlur(gray, 3)                          # apply median filter
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)   # detect edges
    color = cv2.bilateralFilter(fr, int(scale_value.get()), 300, 300)   # apply bilateral filter   
    cartoon = cv2.bitwise_and(color, color, mask=edges)     # combine color image with edges
    stored_frame = cartoon                                  # set the current frame to the cartoonized frame
    if not swap_active:                                     # if swap is not active
        put_frame()                                             # put the frame in the camera widget
        after_id = camera_widget.after(milsec, cartoonize_frame)# call this function again
        
def change_eyes():
    """
    Change the eyes of the current frame
    """
    global stored_frame
    global after_id
    global face_detector
    global shape_predictor
    global eye_active
    global eye
    global swap_active

    remove_filters()                                            # remove all the filters
    eye_active = True                                           # set eye swap as active
    if not swap_active:                                         # if swap is not active
        app.after_cancel(after_id)                                  # stop calling the function
        _, fr = capture.read()                                      # read the current frame
        fr = cv2.flip(fr,1)                                         # flip the frame horizontally
    else:
        fr = stored_frame                                           # use the stored frame
    gray = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)                  # convert the frame to grayscale
    try:
        landmarks_points_frame = detect_facial_landmarks(gray)  # detect the landmarks of the face
        if len(landmarks_points_frame) != 0:                    # if the face is detected
                
                tlel = landmarks_points_frame[37]               # top left eye landmark
                brel = landmarks_points_frame[40]               # bottom left eye landmark

                tler = landmarks_points_frame[43]               # top right eye landmark
                brer = landmarks_points_frame[46]               # bottom right eye landmark

                l_eye_width = abs(brel[0] - tlel[0]  )          # calculate the width of the left eye
                l_eye_height = abs(brel[1] - tlel[1] )          # calculate the height of the left eye

                r_eye_width = abs(brer[0] - tler[0] )           # calculate the width of the right eye
                r_eye_height = abs(brer[1] - tler[1]  )         # calculate the height of the right eye

                eye1 = cv2.resize(eye, (int(l_eye_width), int(l_eye_height)))                # resize the eye image to the width and height of the left eye
                eye_area1 = fr[tlel[1]:tlel[1] + l_eye_height, tlel[0]:tlel[0] + l_eye_width]# get the eye area from the frame
                eye2 = cv2.resize(eye, (int(r_eye_width), int(r_eye_height)))                # resize the eye image to the width and height of the right ey
                eye_area2 = fr[tler[1]:tler[1] + r_eye_height, tler[0]:tler[0] + r_eye_width]# get the eye area from the frame

                left_eye_gray = cv2.cvtColor(eye1, cv2.COLOR_BGR2GRAY)                       # convert the left eye image to grayscale
                _, eye1_mask = cv2.threshold(left_eye_gray, 25, 255, cv2.THRESH_BINARY_INV)  # create a mask for the left eye

                right_eye_gray = cv2.cvtColor(eye2, cv2.COLOR_BGR2GRAY)                      # convert the right eye image to grayscale
                _, eye2_mask = cv2.threshold(right_eye_gray, 25, 255, cv2.THRESH_BINARY_INV) # create a mask for the right eye

                eye_area1_no_eye = cv2.bitwise_and(eye_area1, eye_area1, mask=eye1_mask)     # get the eye area without the eye
                eye_area2_no_eye = cv2.bitwise_and(eye_area2, eye_area2, mask=eye2_mask)     # get the eye area without the eye

                final_eye1 = cv2.add(eye_area1_no_eye, eye1)    # add the eye to the eye area without the eye
                final_eye2 = cv2.add(eye_area2_no_eye, eye2)    # add the eye to the eye area without the eye

                fr[tlel[1]:tlel[1] + l_eye_height, tlel[0]:tlel[0] + l_eye_width] = final_eye1  # add the eye to the frame
                fr[tler[1]:tler[1] + r_eye_height, tler[0]:tler[0] + r_eye_width] = final_eye2  # add the eye to the frame
    except Exception as e:
        # print("Change eyes ",e)
        pass
    stored_frame = fr
    if not swap_active:
        put_frame()
        after_id = camera_widget.after(milsec, change_eyes)         # call this function again

def splash():
    """
    Show the splash screen
    """

    global splash_active
    global after_id
    global swap_active
    global stored_frame
    global img_path
    global eye_active
    global face_detector
    global shape_predictor
    global eye

    remove_filters()                                            # remove all the filters
    splash_active = True                                        # set splash screen as active
    if not swap_active:                                # if the splash screen is not active
        app.after_cancel(after_id)                                  # stop calling the function
        _, fr = capture.read()                                      # read the current frame
        fr = cv2.flip(fr,1)                                         # flip the frame horizontally
    else:
        fr = stored_frame                                           # use the stored frame
    res = np.zeros(fr.shape, np.uint8) # creating blank mask for result
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    #stored_frame = hsv
    # for red
    lower1 = np.array([160,140,20]) # setting lower HSV value
    upper1 = np.array([180,255,255]) # setting upper HSV value
    mask = cv2.inRange(hsv, lower1, upper1) # generating mask

    lower2 = np.array([0,140,20]) # setting lower HSV value
    upper2 = np.array([10,255,255]) # setting upper HSV value
    mask2 = cv2.inRange(hsv, lower2, upper2) # generating mask

    mask = mask + mask2

    # for blue
    # lower1 = np.array([100,100,20]) # setting lower HSV value
    # upper1 = np.array([120,255,255]) # setting upper HSV value
    # mask = cv2.inRange(hsv, lower1, upper1) # generating mask

    #for green and yellow
    # lower1 = np.array([20,0,0]) # setting lower HSV value (20,0,20) (46 0 20)
    # upper1 = np.array([80,255,255]) # setting upper HSV value  (86 255 255)
    # mask = cv2.inRange(hsv, lower1, upper1) # generating mask

    # for yellow
    # lower1 = np.array([20,100,100]) # setting lower HSV value
    # upper1 = np.array([30,255,255]) # setting upper HSV value
    # mask = cv2.inRange(hsv, lower1, upper1) # generating mask

    inv_mask = cv2.bitwise_not(mask) # inverting mask
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    res1 = cv2.bitwise_and(fr, fr, mask= mask) # region which has to be in color
    #stored_frame = res1
    res2 = cv2.bitwise_and(gray, gray, mask= inv_mask) # region which has to be in grayscale
    for i in range(3):
        res[:, :, i] = res2 # storing grayscale mask to all three slices
    fr = cv2.bitwise_or(res1, res) # joining grayscale and color region
    stored_frame = fr
    if not swap_active:
        put_frame()
        after_id = camera_widget.after(milsec, splash)              # call this function again

def pack_scale(from_:int, to:int, text:str) -> None:
    """
    Packs the scale in the GUI
    Args:
        from_ (int): the minimum value of the scale
        to (int): the maximum value of the scale
        text (str): the title of the scale
    """
    scale.config(from_=from_, to=to)        # set the scale range
    scale_title.config(text=text)           # set the scale title
    scale_title.pack(pady=10)               # pack the scale title
    scale.pack(pady=5)                      # pack the scale
    s_value.pack(pady=5)                    # pack the scale value

def stop_filter() -> None:
    """
    Stops the filter
    """
    global img_path  
    app.after_cancel(after_id)              # cancel the after id
    remove_filters()                        # remove all the filters
    try:                         
        if img_path != "":                      # if the image path is not empty
            img = cv2.imread(img_path)          # read the image
            realtime_face_swap(img)             # swap the face
        else:
            open_camera()                       # open the camera
    except Exception as e:
        # print("Stop filter ",e)
        open_camera()
    
def remove_filters():
    """
    Removes all the filters
    """
    global cartoon_active
    global eye_active
    global splash_active
    if cartoon_active:
        cartoon_active = False
        scale.pack_forget()                     # forget the scale
        scale_title.pack_forget()               # forget the scale title
        s_value.pack_forget()                   # forget the scale value
    if eye_active:
        eye_active = False
    if splash_active:
        splash_active = False

    
# ---------- MAIN ------------ #

# CREATE THE CAMERA
cam = Camera(1280,720)                  # create a camera object
capture = cam.record()                  # record video from the camera

# INITIALIZE VARIABLES
milsec = 20                             # the time between each frame
swap_active = False                     # variable to check if the swap is active
cartoon_active = False                  # variable to check if the cartoon filter is active
eye_active = False                      # variable to check if the eye filter is active
splash_active = False                   # variable to check if the splash filter is active
eye = cv2.imread("imgs/blue_eye.png")   # read the eye image
# import detector to detect faces in the image (HOG-based)
face_detector = dlib.get_frontal_face_detector()
# import shape predictor to predict the location of 68 landmarks (points) on the face
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# DRAW THE GUI
app = ttk.Window(themename="minty")                                 # create the GUI
screen_width = app.winfo_screenwidth()                              # get the screen width
screen_height = app.winfo_screenheight()                            # get the screen height
app.geometry(f"{screen_width}x{screen_height}")                     # set the size of the window
app.title("Face Swapper - Camera")                                  # set the title 
app.bind('<Escape>', lambda e: app.quit())                          # press ESC to close the app
app.wm_iconphoto(True, ImageTk.PhotoImage(file="imgs/persona_speciale.png")) # set the icon (da cambiare o togliere dato che è meme)

scale_value = DoubleVar()               # type required by tkinter
scale_value.set(1)                      # starting value

buttons_frame = ttk.Frame(app, padding=10)
buttons_frame.pack(side='left', fill='y', padx=30, pady=50)

swapping_label = ttk.LabelFrame(buttons_frame, text="Face swap", padding=10)
swapping_label.pack()

ttk.Button(swapping_label, text="Upload Photo", width=30, command=upload_image).pack()
ttk.Button(swapping_label, text="Default Camera", width=30, style="warning", command=default_camera).pack(pady=5)

filter_label = ttk.LabelFrame(buttons_frame, text="Filters", padding=10)
filter_label.pack()

cartoonize_button = ttk.Button(filter_label, text="Cartoonize", width=30, command=cartoonize_frame)
cartoonize_button.pack()

eyes_button = ttk.Button(filter_label, text="Change eyes", width=30, command=change_eyes)
eyes_button.pack(pady=5)

splash_button = ttk.Button(filter_label, text="Splash", width=30, command=splash)
splash_button.pack(pady=5)

scale_title = ttk.Label(filter_label, text="scale")
scale_title.pack(pady=10)
scale_title.pack_forget()

scale = ttk.Scale(filter_label, variable=scale_value, length=200, orient='horizontal', from_=1, to=10, command= lambda x: s_value.config(text=int(scale_value.get())))
scale.pack(pady=10)
scale.pack_forget()

s_value = ttk.Label(filter_label, text=int(scale_value.get()))
s_value.pack(pady=10)
s_value.pack_forget()

remove_filter_button = ttk.Button(filter_label, text="Remove filter", width=30, style="warning", command=stop_filter)
remove_filter_button.pack(pady=5)

camera_frame = ttk.Frame(app, padding=10)
camera_frame.pack(side='right')

text_widget = ttk.Label(camera_frame, text="")
text_widget.pack()

camera_label = ttk.LabelFrame(camera_frame, text="Camera", padding=10)
camera_label.pack(padx=10)

camera_widget = ttk.Label(camera_label)
camera_widget.pack()

default_camera()
app.mainloop()              # run the app

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