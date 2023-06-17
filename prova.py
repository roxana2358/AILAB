import dlib
import cv2
import numpy as np 


def extract_index_nparray(nparray):
    index = None
    for n in nparray[0]:
        index = n
        break
    return index

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
img = cv2.imread("gerry.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


faces = face_detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray,face)
    landmarks_points = []
    for n in range(0,68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # cv2.circle(img,(x,y),2,(0,255,0),-1) per vedere i punti
        landmarks_points.append((x,y))


points = np.array(landmarks_points,np.int32)
convexhull = cv2.convexHull(points)
#cv2.imshow("img",img)
#cv2.waitKey(0)
rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmarks_points)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

#cv2.polylines(img,[convexhull],True,(255,0,0),3)
#cv2.imshow("img",img)
#cv2.waitKey(0)

indexes = []
for t in triangles:
    pt1 = (t[0],t[1])
    pt2 = (t[2],t[3])
    pt3 = (t[4],t[5])

    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)

    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)

    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)

    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1,index_pt2,index_pt3]
        indexes.append(triangle)
    



"""
webcam = cv2.VideoCapture(0)
while(True):
    success,frame = webcam.read()
    cv2.imshow("frame",frame)
    k = cv2.waitKey(30)

    # Triangulation destinazion face
    for triangle_index in indexes:
    # Triangulation of the Second face
    tr1_pt1 = landmarks_points2[triangle_index[0]]
    tr1_pt2 = landmarks_points2[triangle_index[1]]
    tr1_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    if k == ord('q'):
        break
        """