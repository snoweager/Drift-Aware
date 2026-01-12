from replay_buffer import build_replay_buffer
from finetune_model import finetune_model

OLD_IMAGES = "data/valid/images"
NEW_IMAGES = "data/stream_day_7/images"
BUFFER_IMAGES = "data/replay_buffer/images"

BASE_MODEL = "models/baseline_yolo7/weights/best.pt"
DATA_YAML = "data/replay_buffer/data.yaml"

def main():
    build_replay_buffer(
        old_data_dir=OLD_IMAGES,
        new_data_dir=NEW_IMAGES,
        buffer_dir=BUFFER_IMAGES
    )

    finetune_model(
        base_model_path=BASE_MODEL,
        data_yaml=DATA_YAML,
        output_dir="models"
    )

if __name__ == "__main__":
    main()
