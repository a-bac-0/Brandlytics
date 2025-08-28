import os
import cv2

def check_dataset(images_dir, labels_dir, classes=None):
    errors = []
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]

    for img_file in images:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # Check if image loads
        img = cv2.imread(img_path)
        if img is None:
            errors.append(f"❌ Cannot read image: {img_path}")
            continue

        # Check if label exists
        if not os.path.exists(label_path):
            errors.append(f"⚠️ Missing label for {img_file}")
            continue

        # Validate label format
        with open(label_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"❌ Bad format in {label_path}, line {i}: {line.strip()}")
                    continue

                cls, x, y, w, h = parts
                try:
                    cls = int(cls)
                    x, y, w, h = map(float, (x, y, w, h))
                except ValueError:
                    errors.append(f"❌ Non-numeric values in {label_path}, line {i}: {line.strip()}")
                    continue

                if classes and (cls < 0 or cls >= len(classes)):
                    errors.append(f"⚠️ Class id {cls} out of range in {label_path}, line {i}")

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"⚠️ Invalid bbox values in {label_path}, line {i}: {line.strip()}")

    if errors:
        print("\n".join(errors))
        print(f"\nFound {len(errors)} issues.")
    else:
        print("✅ Dataset looks good!")

if __name__ == "__main__":
    # Example usage - change these to your dataset paths
    images_dir = "datasets/your_dataset/images/train"
    labels_dir = "datasets/your_dataset/labels/train"
    classes = ["class0", "class1"]  # Optional, set your classes here or leave None

    check_dataset(images_dir, labels_dir, classes)
