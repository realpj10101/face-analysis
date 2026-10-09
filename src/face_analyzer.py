from deepface import DeepFace

class FaceAnalyzer:
    def analyze(self, face): 
        if face.size == 0:
            return None

        try:
            results = DeepFace.analyze(
                img_path=face,
                actions=["age", "gender"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )

            result = results[0]

            return {
                "age": result["age"],
                "gender": result["dominant_gender"]
            }

        except Exception as error:
            print(f"Face analysis failed: {error}")
            return None