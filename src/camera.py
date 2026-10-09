import cv2

from face_detector import FaceDetector
from face_analyzer import FaceAnalyzer


def start_camera():
    detector = FaceDetector()
    analyzer = FaceAnalyzer()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam")
        return

    frame_count = 0
    detected_faces = []

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read frame")
                break

            if frame_count % 30 == 0:
                detected_faces = []

                faces = detector.detect(frame)

                height, width = frame.shape[:2]

                for face in faces:
                    x1 = max(0, face["x1"])
                    y1 = max(0, face["y1"])

                    x2 = min(width, face["x2"])
                    y2 = min(height, face["y2"])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cropped_face = frame[y1:y2, x1:x2]

                    analysis = analyzer.analyze(cropped_face)

                    detected_faces.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "analysis": analysis
                    })

            for face in detected_faces:
                x1 = face["x1"]
                y1 = face["y1"]
                x2 = face["x2"]
                y2 = face["y2"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                analysis = face["analysis"]

                if analysis is not None:
                    age = analysis["age"]
                    gender = analysis["gender"]

                    label = f"Age: {age} | Gender: {gender}"
                else:
                    label = "Face detected"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Face Analysis", frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()