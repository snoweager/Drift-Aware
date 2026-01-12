import random
import os
import shutil

def build_replay_buffer(
    old_data_dir,
    new_data_dir,
    buffer_dir,
    old_ratio=0.7,
    max_samples=300
):
    os.makedirs(buffer_dir, exist_ok=True)

    old_images = os.listdir(old_data_dir)
    new_images = os.listdir(new_data_dir)

    n_old = int(max_samples * old_ratio)
    n_new = max_samples - n_old

    selected_old = random.sample(old_images, min(n_old, len(old_images)))
    selected_new = random.sample(new_images, min(n_new, len(new_images)))

    for img in selected_old:
        shutil.copy(
            os.path.join(old_data_dir, img),
            os.path.join(buffer_dir, img)
        )

    for img in selected_new:
        shutil.copy(
            os.path.join(new_data_dir, img),
            os.path.join(buffer_dir, img)
        )

    print(f"Replay buffer built: {len(selected_old)} old + {len(selected_new)} new samples")
