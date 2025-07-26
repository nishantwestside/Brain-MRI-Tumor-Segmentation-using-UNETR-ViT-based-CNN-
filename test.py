import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from patchify import patchify

from train import load_dataset, create_dir
from unetr_2d import build_unetr_2d  # ✅ Corrected import
from metrics import dice_loss, dice_coef

""" UNETR Configuration """
cf = {}
cf["image_size"] = 256
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Create results directory """
    create_dir("results")

    """ Load dataset """
    dataset_path = "files"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Build and load model weights """
    model = build_unetr_2d(cf)
    model.compile(loss=dice_loss, optimizer='adam', metrics=[dice_coef, "acc"])
    
    model_path = os.path.join("files", "model.h5")
    model.load_weights(model_path)  # ✅ load only weights, not full model

    """ Predict on test data """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x)

        # Read image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
        norm_image = image / 255.0

        # Patchify
        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(norm_image, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32)
        patches = np.expand_dims(patches, axis=0)

        # Read mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask] * 3, axis=-1)

        # Predict
        pred = model.predict(patches, verbose=0)[0]
        pred = np.concatenate([pred] * 3, axis=-1)

        # Save result
        line = np.ones((cf["image_size"], 10, 3)) * 255
        cat_images = np.concatenate([image, line, mask * 255, line, pred * 255], axis=1)
        save_path = os.path.join("results", name)
        cv2.imwrite(save_path, cat_images)
