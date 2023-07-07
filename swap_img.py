# ---------- IMPORT LIBRARIES ---------- #
import dlib
import cv2
import numpy as np

# ---------- FUNCTIONS ---------- #
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
    landmark_points = []
    for p in range(0,68):
        x = landmarks.part(p).x
        y = landmarks.part(p).y
        landmark_points.append((x,y))
    return landmark_points

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

# ---------- MAIN ------------ #
# 1) READ THE IMAGES
img1 = cv2.imread("src/brad_pitt.jpg")            # read the reference image
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # the grayscale image has only one channel in 
                                                    # comparison with the color format so it's easier 
                                                    # to process for the CPU
img2 = cv2.imread("src/edward_norton.jpg")        # read the image to be swapped
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2) CREATE DETECTOR AND PREDICTOR
# import detector to detect faces in the image (HOG-based)
face_detector = dlib.get_frontal_face_detector()
# import shape predictor to predict the location of 68 landmarks (points) on the face
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3) FIND LANDMARKS IN THE FIRST IMAGE AND CREATE A CONVEX HULL
landmark_points1 = detect_facial_landmarks(img1_gray)
# for x,y in landmark_points1:
#     cv2.circle(img1, (x,y), 3, (255,0,0), -1) 
# cv2.imwrite("src/landmarks.jpg", img1)
np_points1 = np.array(landmark_points1,np.int32)
convexhull1 = cv2.convexHull(np_points1)
# cv2.polylines(img1, [convexhull1], True, (255,0,0), 2)
# cv2.imwrite("src/convexhull.jpg", img1)

# 4) FIND LANDMARKS IN THE SECOND IMAGE AND CREATE A CONVEX HULL
landmark_points2 = detect_facial_landmarks(img2_gray)
np_points2 = np.array(landmark_points2,np.int32)
convexhull2 = cv2.convexHull(np_points2)

# 5) FACE SEGMENTATION OF THE FIRST IMAGE INTO TRIANGLES USING DELAUNAY TRIANGULATION
rect = cv2.boundingRect(convexhull1)                    # get the bounding rectangle of the convex hull
# (x,y,w,h) = rect                                        # get the coordinates of the rectangle and draw it on the image
# cv2.rectangle(img1, (x,y), (x+w,y+h), (255,0,0), 2)
# cv2.imwrite("src/rectangle.jpg", img1)
subdiv = cv2.Subdiv2D(rect)                             # create a subdiv2D object with the rectangle
subdiv.insert(landmark_points1)                         # insert the landmark points
triangles = subdiv.getTriangleList()                    # get the list of triangles
triangles = np.array(triangles, dtype=np.int32)         # convert the list to a numpy array

# process each triangle
triangles_indexes = []
for t in triangles:
    pt1 = (t[0],t[1])
    pt2 = (t[2],t[3])
    pt3 = (t[4],t[5])

    # show the triangles on the image
    # cv2.line(img2, pt1, pt2, (255,0,0), 1)
    # cv2.line(img2, pt2, pt3, (255,0,0), 1)
    # cv2.line(img2, pt3, pt1, (255,0,0), 1)

    # use coordinates to find index of the landmark points: where uses the value to find the index of the point in the array
    # the condition returns an array with the indexes of the points that satisfy the condition -> which point it might be
    # axis=1 returns the first element
    # the custom function returns only the value of the index
    index_pt1 = np.where((np_points1 == pt1).all(axis=1))[0][0]
    index_pt2 = np.where((np_points1 == pt2).all(axis=1))[0][0]
    index_pt3 = np.where((np_points1 == pt3).all(axis=1))[0][0]

    # store the triangles in a list
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = (index_pt1,index_pt2,index_pt3)
        triangles_indexes.append(triangle)
# cv2.imwrite("src/triangles2.jpg", img2) 

# remove duplicates
triangles_indexes = list(set(tuple(t) for t in triangles_indexes))

# cv2.imwrite("src/triangles.jpg", img1)

# 6) APPLY THE TRIANAGLES TO THE SECOND IMAGE
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
new_face = np.zeros(img2.shape, np.uint8)               # create a new image with the same size as the second image

toWarp = set()
toWarp.add(2)
toWarp.add(31)
toWarp.add(41)

for indexes in triangles_indexes:
    v1 = indexes[0]
    v2 = indexes[1]
    v3 = indexes[2]

    # triangle in the reference image
    points1, cropped1, _, _, _, _, _ = get_cropped_triangle(img1, landmark_points1, v1, v2, v3)
    
    # triangle in the current frame
    points2, cropped2, cropped2_mask, x2, y2, w2, h2 = get_cropped_triangle(img2, landmark_points2, v1, v2, v3)
    
    # check if the triangle is in the mouth area - no need to warp it
    if (v1 in mouth_points_set and v2 in mouth_points_set and v3 in mouth_points_set) or (v1 in eyes_points_set and v2 in eyes_points_set and v3 in eyes_points_set):
        warped_triangle = cropped2
    # else warp triangle from the reference image to the current frame
    else:
        points_src = np.float32(points1)
        points_dst = np.float32(points2)
        M = cv2.getAffineTransform(points_src, points_dst)
        warped_triangle = cv2.warpAffine(cropped1, M, (w2, h2), flags=cv2.INTER_NEAREST)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped2_mask)
        # if v1 in toWarp and v2 in toWarp and v3 in toWarp:
        #     cv2.imwrite("src/toWarp1.jpg", cropped1)
        #     cv2.imwrite("src/toWarp2.jpg", cropped2)
        #     cv2.imwrite("src/toWarp3.jpg", warped_triangle)

    # apply the transformation to the frame
    area = new_face[y2: y2 + h2, x2: x2 + w2]             # get the area where the triangle will be applied
    area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)                                  # convert it to grayscale
    _, area_mask = cv2.threshold(area_gray, 1, 255, cv2.THRESH_BINARY_INV)              # create mask
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=area_mask) # apply the mask to the warped triangle
    triangle_area = cv2.add(area, warped_triangle)                                      # add the warped triangle to the area
    new_face[y2: y2 + h2, x2: x2 + w2] = triangle_area    # apply the area to the new face
      
# make a mask of the face
face_mask = np.zeros_like(img2_gray)                                # create a black image the same size of the frame
head_mask = cv2.fillConvexPoly(face_mask, convexhull2, 255)         # fill the face with white
# convexhull_mouth = cv2.convexHull(np_points2[60:])                  # get the convex hull of the mouth
# convexhull_left_eye = cv2.convexHull(np_points2[36:42])             # get the convex hull of the left eye
# convexhull_right_eye = cv2.convexHull(np_points2[42:48])            # get the convex hull of the right eye
# head_mask = cv2.fillConvexPoly(head_mask, convexhull_mouth, 0)      # fill the mouth with black
# head_mask = cv2.fillConvexPoly(head_mask, convexhull_left_eye, 0)   # fill the left eye with black
# head_mask = cv2.fillConvexPoly(head_mask, convexhull_right_eye, 0)  # fill the right eye with black
face_mask = cv2.bitwise_not(head_mask)                              # invert the mask (face and mouth black, background white)

# remove the face from the frame
head_noface = cv2.bitwise_and(img2, img2, mask=face_mask)
# cv2.imwrite('src/head_noface.jpg', head_noface)
# cv2.imwrite('src/new_face.jpg', new_face)

# add the new face to the frame
result = cv2.add(head_noface, new_face)
# cv2.imwrite('src/result.jpg', result)

# 7) SEAMLESS CLONING
(x, y, w, h) = cv2.boundingRect(convexhull2)                # get the bounding rectangle of the face
center_face = (int((x + x + w) / 2), int((y + y + h) / 2))  # get the center of the face
seamlessclone = cv2.seamlessClone(result, img2, head_mask, center_face, cv2.MIXED_CLONE)
# cv2.imwrite('src/seamless_mixed.jpg', seamlessclone)

cv2.imshow('Final', seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()