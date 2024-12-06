from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video capture for the webcam
camera = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.degrees(math.acos(cosine_angle))

    return angle

def draw_angle(frame, angle, point, color=(0, 255, 0)):
    """Draws the angle text and an ellipse on the frame."""
    cv2.putText(frame, f'{int(angle)}', tuple(map(int, point[:2])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def calculate_and_draw_angles(frame, landmarks):
    # Extract coordinates for the joints
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]]
    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * frame.shape[0]]

    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame.shape[0]]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame.shape[0]]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * frame.shape[0]]



    # Calculate angles for the joints
    elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
    knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
    hip_angle_left = calculate_angle(shoulder_left, hip_left, knee_left)
    ankle_angle_left = calculate_angle(knee_left, ankle_left, foot_left)
    shoulder_angle_left = calculate_angle(elbow_left, shoulder_left, hip_left)

    elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
    knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
    hip_angle_right = calculate_angle(shoulder_right, hip_right, knee_right)
    ankle_angle_right = calculate_angle(knee_right, ankle_right, foot_right)
    shoulder_angle_right = calculate_angle(elbow_right, shoulder_right, hip_right)


    # Draw the angles on the frame
    draw_angle(frame, elbow_angle_left, elbow_left)
    draw_angle(frame, knee_angle_left, knee_left)
    draw_angle(frame, hip_angle_left, hip_left)
    draw_angle(frame, ankle_angle_left, ankle_left)
    draw_angle(frame, shoulder_angle_left, shoulder_left)

    draw_angle(frame, elbow_angle_right, elbow_right)
    draw_angle(frame, knee_angle_right, knee_right)
    draw_angle(frame, hip_angle_right, hip_right)
    draw_angle(frame, ankle_angle_right, ankle_right)
    draw_angle(frame, shoulder_angle_right, shoulder_right)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Process the frame with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Draw landmarks and calculate angles
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                calculate_and_draw_angles(frame, results.pose_landmarks.landmark)

            # Encode the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
