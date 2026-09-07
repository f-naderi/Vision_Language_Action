import os
import glob
import io
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tfrecord.reader import tfrecord_loader


# LIBERO Dataset
class LiberoDataset(Dataset):

    def __init__(self, episodes, episode_ids, image_size=224, use_wrist_image=False, augment=False):
        self.episodes = episodes
        self.episode_ids = episode_ids
        self.image_size = image_size
        self.use_wrist_image = use_wrist_image
        self.augment = augment

        # ----------------------------------------------------
        # Image transformations
        # ----------------------------------------------------

        if self.augment:

            # Training augmentation
            # Only mild transformations are used because image, state and action must remain consistent.
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor()
            ])

        else:

            # Validation / Test
            # No augmentation.
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

        # ----------------------------------------------------
        # Create timestep samples
        # ----------------------------------------------------

        # Each sample is:  (episode_index, timestep_index)
        # Therefore all timesteps belonging to an episode remain inside the same dataset split.

        self.samples = []

        for episode_idx, episode in enumerate(self.episodes):

            num_steps = len(episode["steps"]["action"])
            for step_idx in range(num_steps):
                self.samples.append((episode_idx, step_idx))

        print(f"Created dataset: {len(self.episodes)} episodes, {len(self.samples)} timesteps")
        print(f"Augmentation: {'ON' if self.augment else 'OFF'}")


    # Dataset length
    def __len__(self):
        return len(self.samples)


    # Image decoding
    def _decode_image(self, image_bytes):

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self.image_transform(image)
        return image


    # Get one timestep
    def __getitem__(self, index):

        episode_idx, step_idx = self.samples[index]
        episode = self.episodes[episode_idx]
        steps = episode["steps"]

        # ----------------------------------------------------
        # Main RGB image
        # ----------------------------------------------------

        image_bytes = (steps["observation"]["image"][step_idx])
        image = self._decode_image(image_bytes)

        # Robot state
        state = torch.tensor(steps["observation"]["state"][step_idx], dtype=torch.float32)

        # Robot action
        action = torch.tensor(steps["action"][step_idx], dtype=torch.float32)

        # Language instruction
        instruction = (steps["language_instruction"][step_idx])

        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

        # Sample
        sample = {"image": image, "state": state, "action": action, "instruction": instruction}

        # Optional wrist camera
        if self.use_wrist_image:
            wrist_image_bytes = (steps["observation"]["wrist_image"][step_idx])
            wrist_image = self._decode_image(wrist_image_bytes)
            sample["wrist_image"] = wrist_image

        return sample



# TFRecord parsing
def load_episodes(data_dir):


    tfrecord_files = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                "libero_spatial-train.tfrecord-*"
            )
        )
    )

    print(f"Found {len(tfrecord_files)} TFRecord files.")

    # TFRecord feature description
    description = {
        "steps/action": "float",
        "steps/observation/state": "float",
        "steps/observation/joint_state": "float",
        "steps/observation/image": "byte",
        "steps/observation/wrist_image": "byte",
        "steps/language_instruction": "byte",
        "steps/reward": "float",
        "steps/discount": "float",
        "steps/is_terminal": "int",
        "steps/is_first": "int",
        "steps/is_last": "int"
    }

    episodes = []

    # Read every TFRecord shard
    for file_path in tfrecord_files:

        loader = tfrecord_loader(file_path, None, description)

        # Read every episode
        for record in loader:

            # Action: [T, 7]
            action = np.asarray(record["steps/action"], dtype=np.float32).reshape(-1, 7)

            # State: [T, 8]
            state = np.asarray(record["steps/observation/state"], dtype=np.float32).reshape(-1, 8)

            # Joint state: [T, 7]
            joint_state = np.asarray(record["steps/observation/joint_state"], dtype=np.float32).reshape(-1, 7)

            # ------------------------------------------------
            # Images
            # ------------------------------------------------

            images = list(record["steps/observation/image"])
            wrist_images = list(record["steps/observation/wrist_image"])


            # Language instructions
            instructions = list(record["steps/language_instruction"])

            # ------------------------------------------------
            # Other RLDS information
            # ------------------------------------------------

            reward = np.asarray(record["steps/reward"], dtype=np.float32)
            discount = np.asarray(record["steps/discount"], dtype=np.float32)
            is_terminal = np.asarray(record["steps/is_terminal"], dtype=np.int64)
            is_first = np.asarray(record["steps/is_first"], dtype=np.int64)
            is_last = np.asarray(record["steps/is_last"], dtype=np.int64)

            # ------------------------------------------------
            # Construct episode
            # ------------------------------------------------

            episode = {
                "steps": {
                    "action": action,
                    "observation": {"image": images, "wrist_image": wrist_images, "state": state, "joint_state": joint_state},
                    "language_instruction": instructions,
                    "reward": reward,
                    "discount": discount,
                    "is_terminal": is_terminal,
                    "is_first": is_first,
                    "is_last": is_last
                }
            }

            episodes.append(episode)

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    print(f"Loaded {len(episodes)} episodes.")

    total_timesteps = sum(
        len(episode["steps"]["action"])
        for episode in episodes
    )

    print(f"Total timesteps: {total_timesteps}")
    return episodes


# Episode-level Train / Validation / Test split
def create_datasets(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, image_size=224, use_wrist_image=False):


    # --------------------------------------------------------
    # Load all episodes
    # --------------------------------------------------------

    episodes = load_episodes(data_dir)
    num_episodes = len(episodes)

    # Create episode IDs
    episode_indices = list(range(num_episodes))

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    rng = random.Random(seed)
    rng.shuffle(episode_indices)

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    num_train = int(num_episodes * train_ratio)
    num_val = int(num_episodes * val_ratio)

    # Remaining episodes go to Test.
    num_test = (num_episodes - num_train - num_val)

    # --------------------------------------------------------
    # Split episode IDs
    # --------------------------------------------------------

    train_ids = episode_indices[:num_train]
    val_ids = episode_indices[num_train:num_train + num_val]
    test_ids = episode_indices[num_train + num_val:]

    # --------------------------------------------------------
    # Create episode subsets
    # --------------------------------------------------------

    train_episodes = [
        episodes[i]
        for i in train_ids
    ]

    val_episodes = [
        episodes[i]
        for i in val_ids
    ]

    test_episodes = [
        episodes[i]
        for i in test_ids
    ]

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    print("\nEpisode-level split:")
    print(f"Train: {len(train_episodes)} episodes")
    print(f"Val:   {len(val_episodes)} episodes")
    print(f"Test:  {len(test_episodes)} episodes")

    # Create Train dataset
    train_dataset = LiberoDataset(train_episodes, train_ids, image_size=image_size, use_wrist_image=use_wrist_image, augment=True)

    # Create Validation dataset
    val_dataset = LiberoDataset(val_episodes, val_ids, image_size=image_size, use_wrist_image=use_wrist_image, augment=False)

    # Create Test dataset
    test_dataset = LiberoDataset(test_episodes, test_ids, image_size=image_size, use_wrist_image=use_wrist_image, augment=False)

    return (train_dataset, val_dataset, test_dataset)