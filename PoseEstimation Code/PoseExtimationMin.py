import math

import cv2
import mediapipe as mp
import time

import numpy as np


def calculateAngle(landmark1,landmark2,landmark3):

    #get req cordinates
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    #calculate angle
    angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

    if angle<0:
        angle+=360
    return angle


def MAPE(data_arr, real_arr):
    return np.mean(abs((data_arr - real_arr) / real_arr) * 100)


# Initialize Mediapipe components
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load the input image
ideal = cv2.imread("vrkshasana.jpg")  # Fixed the image loading
idealRGB = cv2.cvtColor(ideal, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Process the image to detect pose landmarks
idealresult = pose.process(idealRGB)  # Fixed variable name

# Check if pose landmarks were detected
if idealresult.pose_landmarks:
    # Draw pose landmarks and connections on the image
    #mpDraw.draw_landmarks(ideal, idealresult.pose_landmarks, mpPose.POSE_CONNECTIONS)
    real=[]
    idealpt=[]
    # Iterate through landmarks and get their positions
    hi, wi, ci = ideal.shape
    for id, lm in enumerate(idealresult.pose_landmarks.landmark):
        cxi, cyi, czi = int(lm.x * wi), int(lm.y * hi),int(lm.z * wi)
        idealpt.append((cxi, cyi, czi))

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                      idealpt[mpPose.PoseLandmark.LEFT_ELBOW.value],
                                      idealpt[mpPose.PoseLandmark.LEFT_WRIST.value])
    real.append(left_elbow_idealangle)
    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                       idealpt[mpPose.PoseLandmark.RIGHT_ELBOW.value],
                                       idealpt[mpPose.PoseLandmark.RIGHT_WRIST.value])
    real.append(right_elbow_idealangle)
    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.LEFT_ELBOW.value],
                                         idealpt[mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                         idealpt[mpPose.PoseLandmark.LEFT_HIP.value])
    real.append(left_shoulder_idealangle)
    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.RIGHT_HIP.value],
                                          idealpt[mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                          idealpt[mpPose.PoseLandmark.RIGHT_ELBOW.value])
    real.append(right_shoulder_idealangle)
    # Get the angle between the left hip, knee and ankle points.
    left_knee_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.LEFT_HIP.value],
                                     idealpt[mpPose.PoseLandmark.LEFT_KNEE.value],
                                     idealpt[mpPose.PoseLandmark.LEFT_ANKLE.value])
    real.append(left_knee_idealangle)
    # Get the angle between the right hip, knee and ankle points
    right_knee_idealangle = calculateAngle(idealpt[mpPose.PoseLandmark.RIGHT_HIP.value],
                                      idealpt[mpPose.PoseLandmark.RIGHT_KNEE.value],
                                      idealpt[mpPose.PoseLandmark.RIGHT_ANKLE.value])
    real.append(right_knee_idealangle)
    real_arr=np.array(real)
    # ----------------------------------------------------------------------------------------------------------------


cv2.imshow('Mediapipe feed', cv2.flip(ideal,1))
cv2.waitKey(0)



cap = cv2.VideoCapture(0)

pTime=0
while cap.isOpened():
    ret , img = cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)

    if results.pose_landmarks:
        #mpDraw.draw_landmarks(img,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h,w,c=img.shape
        data=[]
        pt=[]
        for id,lm in enumerate(results.pose_landmarks.landmark) :
            cx, cy, cz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
            pt.append((cx, cy, cz))
        # Calculate the required angles.
        # ----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left shoulder, elbow and wrist points.
        left_elbow_angle = calculateAngle(pt[mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                          pt[mpPose.PoseLandmark.LEFT_ELBOW.value],
                                          pt[mpPose.PoseLandmark.LEFT_WRIST.value])
        data.append(left_elbow_angle)
        # Get the angle between the right shoulder, elbow and wrist points.
        right_elbow_angle = calculateAngle(pt[mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                           pt[mpPose.PoseLandmark.RIGHT_ELBOW.value],
                                           pt[mpPose.PoseLandmark.RIGHT_WRIST.value])
        data.append(right_elbow_angle)
        # Get the angle between the left elbow, shoulder and hip points.
        left_shoulder_angle = calculateAngle(pt[mpPose.PoseLandmark.LEFT_ELBOW.value],
                                             pt[mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                             pt[mpPose.PoseLandmark.LEFT_HIP.value])
        data.append(left_shoulder_angle)
        # Get the angle between the right hip, shoulder and elbow points.
        right_shoulder_angle = calculateAngle(pt[mpPose.PoseLandmark.RIGHT_HIP.value],
                                              pt[mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                              pt[mpPose.PoseLandmark.RIGHT_ELBOW.value])
        data.append(right_shoulder_angle)
        # Get the angle between the left hip, knee and ankle points.
        left_knee_angle = calculateAngle(pt[mpPose.PoseLandmark.LEFT_HIP.value],
                                         pt[mpPose.PoseLandmark.LEFT_KNEE.value],
                                         pt[mpPose.PoseLandmark.LEFT_ANKLE.value])
        data.append(left_knee_angle)
        # Get the angle between the right hip, knee and ankle points
        right_knee_angle = calculateAngle(pt[mpPose.PoseLandmark.RIGHT_HIP.value],
                                          pt[mpPose.PoseLandmark.RIGHT_KNEE.value],
                                          pt[mpPose.PoseLandmark.RIGHT_ANKLE.value])
        data.append(right_knee_angle)
        data_arr=np.array(data)
        # ----------------------------------------------------------------------------------------------------------------


        mape = MAPE(data_arr, real_arr)
        acc=round(100-mape,2)

        if(acc>80):
            cv2.putText(img,("Accuracy :"+ str(acc))+ " %", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(img, "Please do the pose Properly And be visible in screen", (25, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)



    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #cv2.putText(img,str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,0) , 3)

    cv2.imshow('Mediapipe feed', img ) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()