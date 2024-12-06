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
                # Extract landmarks for shoulder, elbow, wrist, hip, knee, and ankle
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]]

                # Calculate angles for important joints
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                # Example for other angles like hip flexion if needed

                # Draw a semi-circle at the elbow
                center_elbow = tuple(map(int, elbow))
                axes = (100, 100)
                angle_start = 180
                angle_end = angle_start + int(elbow_angle)
                color = (0, 255, 0)
                cv2.ellipse(frame, center_elbow, axes, 0, angle_start, angle_end, color, 2)

                # Draw a semi-circle at the knee
                center_knee = tuple(map(int, knee))
                angle_start = 180
                angle_end = angle_start + int(knee_angle)
                cv2.ellipse(frame, center_knee, axes, 0, angle_start, angle_end, color, 2)

                # Display the angles on the frame
                cv2.putText(frame, f'Elbow: {int(elbow_angle)}°', tuple(map(int, elbow)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Knee: {int(knee_angle)}°', tuple(map(int, knee)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Encode the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
