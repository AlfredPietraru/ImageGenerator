import os
import cv2
import torch
from torchvision.transforms.functional import rotate


INITIAL_IMAGES_PATH = './images/train/initial_images/'
GROUNDTRUTH_IMAGE_PATH = './images/train/groundtruth/'
MODIFIED_TRAINING_IMAGES_PATH = './images/train/modified/'

TEST_GROUND_TRUTH = './images/test/groundtruth/'
TEST_INITIAL_IMAGES = './images/test/initial_images/'
TEST_MODIFIED_IMAGES = './images/test/modified/'

SIZE = 64
def map_int_to_string(idx: int):
    return f"{idx:03d}.png"

def transformers(img : torch.tensor):
  return torch.cat([rotate(img, angle=90),
      rotate(img, angle=180), rotate(img, angle=270)], dim=0)

def create_initial_images(input_folder, output_folder, size=SIZE):
    for idx, filename in enumerate(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (SIZE, SIZE))
        output_path = os.path.join(output_folder, map_int_to_string(idx))
        cv2.imwrite(output_path, resized_img)
        
def show_image(img : torch.tensor):
    img = img.to(torch.uint8).permute((1, 2, 0))
    img = img.detach().numpy()
    img = cv2.resize(img, (256, 256))
    cv2.imshow("gata", img)
    cv2.waitKey(0)

def get_only_training_image(batch_index : int, batch_size):
    images_indexes = range(batch_index * batch_size, (batch_index + 1) * batch_size) 
    filenames = list(map(map_int_to_string, images_indexes))
    images = torch.zeros(size=(batch_size, 3, SIZE, SIZE))
    for idx, filename in enumerate(filenames):
        img = cv2.imread(INITIAL_IMAGES_PATH + filename, 
                                cv2.IMREAD_COLOR)
        images[idx] = torch.tensor(img).permute((2, 0, 1))
    return images


def get_one_training_image(idx : int):
    filename = map_int_to_string(idx)
    img = cv2.imread(INITIAL_IMAGES_PATH + filename, 
                                cv2.IMREAD_COLOR)
    img = torch.tensor(img).permute((2, 0, 1))
    return img

if __name__ == '__main__':
    create_initial_images(GROUNDTRUTH_IMAGE_PATH, INITIAL_IMAGES_PATH)