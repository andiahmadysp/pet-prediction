from PIL import Image
import os

dataset_path = "dataset/PetImages"

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