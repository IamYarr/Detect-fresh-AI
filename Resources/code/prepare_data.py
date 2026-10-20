import shutil
import random
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "Resources" / "raw_img"
TARGET_DATA_DIR = BASE_DIR / "Resources" / "img"

CLASSES = ["FRESH", "MINOR_DEFECT", "SPOILED"]
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

def process_and_move_data():
    if not RAW_DATA_DIR.exists():
        print(f"Folder {RAW_DATA_DIR} does not exist. Creating one...")
        for cls in CLASSES:
            (RAW_DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
        return
    if not TARGET_DATA_DIR.exists():
        print(f"Folder {TARGET_DATA_DIR} does not exist. Creating one...")
        for cls in CLASSES:
            (TARGET_DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
        return

    for class_name in CLASSES:
        class_dir = RAW_DATA_DIR / class_name
        if not class_dir.exists():
            print(f"Warning! {class_dir} folder does not exist!")
            continue

        images = [f for f in class_dir.iterdir() if f.is_file()]
        total_images = len(images)
        
        if total_images == 0:
            print(f"{class_dir} Warning! No images were selected!")
            continue
        random.shuffle(images)

        train_end = int(total_images * SPLIT_RATIO["train"])
        val_end = train_end + int(total_images * SPLIT_RATIO["val"])

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_images in splits.items():
            dest_dir = TARGET_DATA_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                unique_id = uuid.uuid4().hex[:8]
                # Генерация уникального имени для защиты от перезаписи
                new_filename = f"{img_path.stem}_{unique_id}{img_path.suffix}"
                dest_path = dest_dir / new_filename
                shutil.move(str(img_path), str(dest_path))
                
        print(f"Class {class_name}: added {total_images} new photos"
              f"(Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")

if __name__ == "__main__":
    process_and_move_data()
    print("Images proccesed")
