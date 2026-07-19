from pathlib import Path

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "CE_Xiphoid_Project"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "external_test_dataset"
    / "images"
    / "test"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "external_test_dataset"
    / "labels"
    / "test"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

image_stems = {
    path.stem
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
}

label_stems = {
    path.stem
    for path in LABEL_DIR.glob("*.txt")
}

images_without_labels = sorted(
    image_stems - label_stems
)

labels_without_images = sorted(
    label_stems - image_stems
)

print("=" * 60)
print("EXTERNAL TEST DATASET CHECK")
print("=" * 60)

print(f"Images: {len(image_stems)}")
print(f"Labels: {len(label_stems)}")

print()
print(
    "Images without labels:",
    len(images_without_labels),
)

for name in images_without_labels[:20]:
    print("  ", name)

print()
print(
    "Labels without images:",
    len(labels_without_images),
)

for name in labels_without_images[:20]:
    print("  ", name)

print()
print("Expected:")
print("  Images: 214")
print("  Labels: 214")
print("  Images without labels: 0")
print("  Labels without images: 0")

print("=" * 60)