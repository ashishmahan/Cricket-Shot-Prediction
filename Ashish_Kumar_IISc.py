import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to extract keypoints using Mediapipe
def extract_keypoints(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image=img_rgb)
        
        if results.pose_landmarks is None:
             return None
        
         # Extract pose landmarks
        pose_landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y] for lm in pose_landmarks])
        return keypoints.flatten()



def extract_and_visualize_keypoints(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image=img_rgb)
        
        if results.pose_landmarks is None:
            return img  
        
       
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return img

def visualize_images(dataset_dir, shot_types, clf):
    for shot_type in shot_types:
        print(f"Visualizing images for class: {shot_type}")
        shot_dir = os.path.join(dataset_dir, shot_type)
        image_files = os.listdir(shot_dir)
        
        for i, image_file in enumerate(image_files[:3]):
            image_path = os.path.join(shot_dir, image_file)
            visualized_image = extract_and_visualize_keypoints(image_path)
            cv2.imshow(f"{shot_type} - Image {i+1}", visualized_image)
            cv2.waitKey(0)  
            # Predict shot type and overlay on image
            predicted_class = shot_types[clf.predict([extract_keypoints(image_path)])[0]]
            visualized_image = cv2.putText(visualized_image, f"{predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"{shot_type} - Image {i+1}", visualized_image)
            cv2.waitKey(0)  
        cv2.destroyAllWindows()


def predict_shot_type(image_path, clf, shot_types):
    image = cv2.imread(image_path)
    keypoints = extract_keypoints(image_path)
    
    if keypoints is not None:
        predicted_class = shot_types[clf.predict([keypoints])[0]]
        annotated_image = extract_and_visualize_keypoints(image)
        annotated_image = cv2.putText(annotated_image, f"{predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Predicted shot type:", predicted_class)
    else:
        print("No pose landmarks detected in the image.")


if __name__ == "__main__":
    dataset_dir = r"C:\Users\ashis\Desktop\data"
    shot_types = ["drive", "legglance-flick", "pullshot", "sweep"]

    X, y = [], []
    for idx, shot_type in enumerate(shot_types):
        shot_dir = os.path.join(dataset_dir, shot_type)
        for image_file in os.listdir(shot_dir):
            image_path = os.path.join(shot_dir, image_file)
            keypoints = extract_keypoints(image_path)
            if keypoints is not None:
                X.append(keypoints)
                y.append(idx)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("Classification Report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=shot_types))

    visualize_images(dataset_dir, shot_types, clf)

    image_path1 = r"C:\Users\ashis\Desktop\marnus.jpeg"
    predict_shot_type(image_path1, clf, shot_types)
