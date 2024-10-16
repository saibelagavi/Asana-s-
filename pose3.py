import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance as dist
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(a,b,c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle

def compare_pose(image, angle_point, angle_user, angle_target):
    angle_user = np.array(angle_user)
    angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    cv2.rectangle(image, (0,0), (370,40), (255,255,255), -1)
    cv2.rectangle(image, (0,40), (370,370), (255,255,255), -1)
    cv2.putText(image, str("Score:"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
    height, width, _ = image.shape   

    if angle_user[0] < (angle_target[0] - 15):
        stage += 1
        cv2.putText(image, str("Extend the right arm at elbow"), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0]*width), int(angle_point[0][1]*height)), 30, (0,0,255), 5) 
        
    if angle_user[0] > (angle_target[0] + 15):
        stage += 1
        cv2.putText(image, str("Fold the right arm at elbow"), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0]*width), int(angle_point[0][1]*height)), 30, (0,0,255), 5)
        
    if angle_user[1] < (angle_target[1] -15):
        stage += 1
        cv2.putText(image, str("Extend the left arm at elbow"), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0]*width), int(angle_point[1][1]*height)), 30, (0,0,255), 5)
        
    if angle_user[1] > (angle_target[1] + 15):
        stage += 1
        cv2.putText(image, str("Fold the left arm at elbow"), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0]*width), int(angle_point[1][1]*height)), 30, (0,0,255), 5)

    if angle_user[2] < (angle_target[2] - 15):
        stage += 1
        cv2.putText(image, str("Lift your right arm"), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0]*width), int(angle_point[2][1]*height)), 30, (0,0,255), 5)

    if angle_user[2] > (angle_target[2] + 15):
        stage += 1
        cv2.putText(image, str("Put your arm down a little"), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0]*width), int(angle_point[2][1]*height)), 30, (0,0,255), 5)

    if angle_user[3] < (angle_target[3] - 15):
        stage += 1
        cv2.putText(image, str("Lift your left arm"), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0]*width), int(angle_point[3][1]*height)), 30, (0,0,255), 5)

    if angle_user[3] > (angle_target[3] + 15):
        stage += 1
        cv2.putText(image, str("Put your arm down a little"), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0]*width), int(angle_point[3][1]*height)), 30, (0,0,255), 5)

    if angle_user[4] < (angle_target[4] - 15):
        stage += 1
        cv2.putText(image, str("Extend the angle at right hip"), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0]*width), int(angle_point[4][1]*height)), 30, (0,0,255), 5)

    if angle_user[4] > (angle_target[4] + 15):
        stage += 1
        cv2.putText(image, str("Reduce the angle of at right hip"), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0]*width), int(angle_point[4][1]*height)), 30, (0,0,255), 5)

    if angle_user[5] < (angle_target[5] - 15):
        stage += 1
        cv2.putText(image, str("Extend the angle at left hip"), (10,260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0]*width), int(angle_point[5][1]*height)), 30, (0,0,255), 5)
        
    if angle_user[5] > (angle_target[5] + 15):
        stage += 1
        cv2.putText(image, str("Reduce the angle at left hip"), (10,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0]*width), int(angle_point[5][1]*height)), 30, (0,0,255), 5)

    if angle_user[6] < (angle_target[6] - 15):
        stage += 1
        cv2.putText(image, str("Extend the angle of right knee"), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0]*width), int(angle_point[6][1]*height)), 30, (0,0,255), 5)
        
    if angle_user[6] > (angle_target[6] + 15):
        stage += 1
        cv2.putText(image, str("Reduce the angle at right knee"), (10,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0]*width), int(angle_point[6][1]*height)), 30, (0,0,255), 5)

    if angle_user[7] < (angle_target[7] - 15):
        stage += 1
        cv2.putText(image, str("Extend the angle at left knee"), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0]*width), int(angle_point[7][1]*height)), 30, (0,0,255), 5)

    if angle_user[7] > (angle_target[7] + 15):
        stage += 1
        cv2.putText(image, str("Reduce the angle at left knee"), (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0]*width), int(angle_point[7][1]*height)), 30, (0,0,255), 5)
    
    if stage != 0:
        cv2.putText(image, str("FIGHTING!"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)       
    else:
        cv2.putText(image, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

def dif_compare(x, y):
    average = []
    for i in range(len(x)):
        result = 1 - dist.cosine(list(x[i].values()), list(y[i].values()))
        average.append(result)
    score = math.sqrt(2 * (1 - round(sum(average) / len(average), 2)))
    return score

def diff_compare_angle(x, y):
    new_x = []
    for i in range(len(x)):
        z = np.abs(x[i] - y[i]) / ((x[i] + y[i]) / 2)
        new_x.append(z)
    return np.mean(new_x)

def convert_data(landmarks):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'vis'])
    for point in landmarks:
        df = df.append({"x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "vis": point.visibility
                        }, ignore_index=True)
    return df

def extractKeypoint(path):
    IMAGE_FILES = [path] 
    stage = None
    joint_list_video = pd.DataFrame([])
    count = 0
    keypoints = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)   
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_h, image_w, _ = image.shape

            try:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Your code for rendering annotations
                
                joints = []
                joint_list = pd.DataFrame([])

                # for i, data_point in enumerate(landmarks):
                #     joints = pd.DataFrame({
                #         'frame': count,
                #         'id': i,
                #         'x': data_point.x,
                #         'y': data_point.y,
                #         'z': data_point.z,
                #         'vis': data_point.visibility
                #     }, index=[0])
                #     joint_list = joint_list.append(joints, ignore_index=True)
                
                for point in landmarks:
                    keypoints.append({
                         'X': point.x,
                         'Y': point.y,
                         'Z': point.z,
                    })

               
                angle = []
                angle_list = pd.DataFrame([])
                angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                angle.append(int(angle1))
                angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                angle.append(int(angle2))
                angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                angle.append(int(angle3))
                angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                angle.append(int(angle4))
                angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                angle.append(int(angle5))
                angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                angle.append(int(angle6))
                angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                angle.append(int(angle7))
                angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                angle.append(int(angle8))
                
                # Your code for rendering annotations

            except Exception as e:
                print("Error:", e)
                pass
            
            joint_list_video = pd.concat([joint_list_video, joint_list], ignore_index=True)


            # Your code for rendering annotations

            # Your code for rendering detections

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()
    return landmarks, keypoints, angle, image


# Main Code
cap = cv2.VideoCapture(0)
path = "D:/Hospital3/Hospital2/Hospital/patient/static/images/tadasana.jpg"
# Assuming extractKeypoint function returns required values
x = extractKeypoint(path)
dim = (960, 760)
resized = cv2.resize(x[3], dim, interpolation=cv2.INTER_AREA)
cv2.imshow('target', resized)
angle_target = x[2]
point_target = x[1]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extracting necessary landmarks
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                        round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility*100, 2)]
            
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility*100, 2)]
            
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility*100, 2)]
            
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            keypoints = []
            for point in landmarks:
                keypoints.append({
                    'X': point.x,
                    'Y': point.y,
                    'Z': point.z,
                })
            
            p_score = dif_compare(keypoints, point_target)
            
            angle = []
            angle.append(calculateAngle(right_shoulder, right_elbow, right_wrist))
            angle.append(calculateAngle(left_shoulder, left_elbow, left_wrist))
            angle.append(calculateAngle(right_elbow, right_shoulder, right_hip))
            angle.append(calculateAngle(left_elbow, left_shoulder, left_hip))
            angle.append(calculateAngle(right_shoulder, right_hip, right_knee))
            angle.append(calculateAngle(left_shoulder, left_hip, left_knee))
            angle.append(calculateAngle(right_hip, right_knee, right_ankle))
            angle.append(calculateAngle(left_hip, left_knee, left_ankle))
            
            compare_pose(image, [right_elbow, left_elbow, right_shoulder, left_shoulder,
                                 right_hip, left_hip, right_knee, left_knee], angle, angle_target)
            
            a_score = diff_compare_angle(angle, angle_target)
            
            if (p_score >= a_score):
                cv2.putText(image, str(int((1 - a_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str(int((1 - p_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
 
        except Exception as e:
            print(f"Error: {e}")

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=4, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3))

        cv2.imshow('MediaPipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
