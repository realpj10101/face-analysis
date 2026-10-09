from pathlib import Path

import cv2
from ultralytics import YOLO

class FaceDetector: 
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "yolo26n-face.pt"

        self.model = YOLO(str(model_path))

    def detect(self, frame): 
        results = self.model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        faces = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                faces.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence
                })

        return faces
