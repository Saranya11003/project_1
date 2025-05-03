import os
import base64
import json
import random
from locust import HttpUser, task, between

class CloudPoseUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        img_dir = "./images"
        self.encoded_images = []
        for fname in os.listdir(img_dir):
            if fname.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(img_dir, fname)
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    self.encoded_images.append(b64)
        
        if not self.encoded_images:
            raise ValueError("No images found in the specified directory.")

    @task(3)
    def pose_json(self):
        img = random.choice(self.encoded_images)
        payload = {
            "id": str(random.randint(1, 1000)),
            "image": img
        }

        self.client.post(
            "/api/pose/json",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            name="POST /api/pose/json"
        )
    
    @task(1)
    def pose_annotated(self):
        img = random.choice(self.encoded_images)
        payload = {
            "id": str(random.randint(1, 1000)),
            "image": img
        }

        self.client.post(
            "/api/pose/annotated",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            name="POST /api/pose/annotated"
        )