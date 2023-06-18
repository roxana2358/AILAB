# ---------- IMPORT LIBRARIES ---------- #
import dlib
import cv2
import numpy as np 


# ---------- CUSTOM FUNCTIONS ---------- #
def extract_index_nparray(nparray):
    index = None
    # return the first element of the array
    for num in nparray[0]:
        index = num
        break
    return index


# ---------- MAIN ------------ #
# 1) READ THE REFERENCE IMAGE
img = cv2.imread("imgs/gerry.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # the grayscale image has only one channel in 
                                                    # comparison with the color format so it's easier 
                                                    # to process for the CPU


# 2) CREATE DETECTOR AND PREDICTOR
# import detector to detect faces in the image (HOG-based)
face_detector = dlib.get_frontal_face_detector()
# import shape predictor to predict the location of 68 landmarks (points) on the face
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# 3) FIND LANDMARKS IN THE REFERENCE IMAGE AND CREATE A CONVEX HULL
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


# 4) FACE SEGMENTATION INTO TRIANGLES USING DELAUNAY TRIANGULATION
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
    
# for the next steps we need the frame from the webcam
webcam = cv2.VideoCapture(0)
while(True):
    _, frame = webcam.read()                            # read the current frame
    frame = cv2.flip(frame,1)                           # flip the frame horizontally
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert the frame to grayscale
    new_face = np.zeros_like(frame)     # create a new image with the same size as the frame

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
    np_points_frame = np.array(landmark_points_frame,np.int32)
    convexhull_frame = cv2.convexHull(np_points_frame) 

# 6) APPLY TRIANGLES TO THE CURRENT FRAME IF THERE ARE LANDMARKS
    if (len(landmark_points_frame) != 0):
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
            cropped_triangle_frame = frame[y_frame: y_frame + h_frame, x_frame: x_frame + w_frame]
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
                
        # swap faces
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

        # show the frame
        cv2.imshow("Final result", seamlessclone)
        k = cv2.waitKey(30)
        if k == ord('q'):       # if the 'q' key is pressed, stop the loop
            break

webcam.release()
cv2.destroyAllWindows()

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