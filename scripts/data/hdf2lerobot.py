import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-r", "--robot", type=str, required=True)
parser.add_argument("-t", "--task", type=str, required=True)
parser.add_argument("-s", "--image_size", type=int, required=False, default=224)
args = parser.parse_args()

import os
os.environ["HF_LEROBOT_HOME"] =f"{args.output}"

import shutil
import h5py 
import numpy as np
from tqdm import tqdm
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from iiil.tasks import TASK_INSTRUCTION_DICT
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def main(args):
    repo_id = f"i3l/{args.task}"
    output_dir = os.path.join(args.output, repo_id)
    if args.input.endswith('hdf5'):
        raw_datasets = [args.input]
    else:
        raw_datasets = sorted([os.path.join(args.input, file_name) for file_name in os.listdir(args.input)])
    # Clean up any existing dataset in the output directory
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Are you sure you want to overwrite it? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting without overwriting the output directory.")
            return
        shutil.rmtree(output_dir)
    print(f"Converting datasets from {args.input} to LeRobot format at {output_dir}")

    action_dim = 7 if args.robot == "A1" else 23
    if args.robot == "A1":
        proprio_dim = 52
    elif args.robot == "R1Pro":
        proprio_dim = 256
    elif args.robot == "RealR1Pro":
        proprio_dim = 46
    else:
        raise NotImplementedError(f"Robot {args.robot} not supported.")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (proprio_dim,),
            "names": ["observation.state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "int_state": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["int_state"],
        },
    }
    for camera in ROBOT_CAMERA_NAMES[args.robot]:
        features[f"observation.rgb.{camera}"] = {
            "dtype": "video",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channel"],
        }

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=args.robot,
        fps=30,
        features=features,
        image_writer_threads=20,
        image_writer_processes=10,
        batch_encoding_size=50,
    )

    # Loop over raw H5 datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset in raw_datasets:
        # get all the tasks that are collected that day 
        print("Processing file: ", raw_dataset)
        with h5py.File(raw_dataset, "r") as raw_data:
            # get the number of demos
            num_demos = len(raw_data["data"].keys())
            for idx in tqdm(range(num_demos)):
                demo_id = f'demo_{idx}'
                demo_data = raw_data["data"][demo_id]
                
                # get the proprio data
                proprio_data = demo_data["obs/robot::proprio"][:]
                int_state = demo_data["policy/int_state"][:]
                # get action
                raw_action = demo_data["action"][:]
                seq_length = proprio_data.shape[0]
                
                resized_images = {
                    camera_name: resize_with_pad(
                        demo_data["obs"][f"{key}::rgb"][...,:3], args.image_size, args.image_size
                    ) for camera_name, key in ROBOT_CAMERA_NAMES[args.robot].items()
                }

                for step in range(seq_length):
                    images_t = {
                        f"observation.rgb.{key}": resized_images[key][step] for key in resized_images
                    }
                    dataset.add_frame(
                        frame={
                            "observation.state": proprio_data[step],
                            "action": raw_action[step],
                            "int_state": int_state[step],
                            **images_t
                        },
                        task=TASK_INSTRUCTION_DICT[args.task],
                    )
                dataset.save_episode()

    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main(args)
