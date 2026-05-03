from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "data", "PetImages")

deleted = 0

for label in ["Cat", "Dog"]:
    folder = os.path.join(dataset_path, label)

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            os.remove(path)
            deleted += 1
            print("deleted:", path)

print("total deleted:", deleted)