from ultralytics import YOLO
import torch

def finetune_model(
    base_model_path,
    data_yaml,
    output_dir,
    epochs=10
):
    model = YOLO(base_model_path)

    # Freeze backbone layers
    for name, param in model.model.named_parameters():
        if "model.0" in name or "model.1" in name:
            param.requires_grad = False

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=4,
        workers=1,
        lr0=1e-4,
        name="incremental_model",
        project=output_dir
    )

    print("Incremental fine-tuning complete.")
