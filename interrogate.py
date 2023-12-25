"""
WD14 Tagger for classifying images
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
import os
import csv
load_model_hf = tf.keras.models.load_model
from tqdm import tqdm
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # set memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = 448
INTERPOLATION = cv2.INTER_AREA
REPOSITORY = "SmilingWolf/wd-v1-4-moat-tagger-v2" #moat or etc 
DTYPE = np.float16
# global params, models, general_tags, character_tags, rating_tags
model:tf.keras.models.Model = None
general_tags:list|None = None
character_tags:list|None = None

def read_tags(base_path):
    """
    Reads tags from selected_tags.csv, and stores them in global variables
    base_path: base path to model (str)
    return: None
    """
    global general_tags, character_tags
    if general_tags is not None and character_tags is not None:
        return None
    with open(os.path.join(base_path, 'selected_tags.csv'), "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        tags = list(reader)
        header = tags.pop(0)
        tags = tags[1:]
    assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"header is not correct for {base_path} selected_tags.csv"
    # if category is 0, general, 4, character, else ignore
    general_tags = [tag[1] for tag in tags if tag[2] == '0']
    character_tags = [tag[1] for tag in tags if tag[2] == '4']
    return None

def preprocess_image(image:Image.Image) -> np.ndarray:
    global IMAGE_SIZE, INTERPOLATION
    # handle RGBA
    assert isinstance(image, Image.Image), f"Expected image to be Image.Image, got {type(image)} with value {image}"
    if image.mode == "RGBA":
        # paste on white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        image = background
    image = np.array(image)
    image = image[:, :, ::-1].copy() # RGB to BGR
    # pad to square image
    target_size = [max(image.shape)] * 2
    # pad with 255 to make it white
    image_padded = 255 * np.ones((target_size[0], target_size[1], 3), dtype=np.uint8)
    dw = int((target_size[0] - image.shape[1]) / 2)
    dh = int((target_size[1] - image.shape[0]) / 2)
    image_padded[dh:image.shape[0]+dh, dw:image.shape[1]+dw, :] = image
    image = image_padded
    # assert
    assert image.shape[0] == image.shape[1]
    
    # resize
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=INTERPOLATION)
    image = image.astype(DTYPE)
    return image

def download_model(repo_dir: str = REPOSITORY, save_dir: str = "./", force_download: bool = False):
    # tagger follows following files
    print("Downloading model")
    FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
    SUB_DIR = "variables"
    SUB_DIR_FILES = [f"{SUB_DIR}.data-00000-of-00001", f"{SUB_DIR}.index"]
    if os.path.exists(save_dir) and not force_download:
        return os.path.abspath(save_dir)
    # download
    for file in FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, file, cache_dir = save_dir, force_download = force_download, force_filename = file)
    for file in SUB_DIR_FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, (SUB_DIR+'/'+ file), cache_dir = os.path.join(save_dir, SUB_DIR), force_download = force_download, force_filename = file)
    return os.path.abspath(save_dir)

# check if model is already loaded in port 5050, if it is, use api
def check_model_loaded():
    """
    Checks if model is loaded in port_number
    port_number: port number to check (int)
    ip: ip address to check (str) default: localhost
    return: True if model is loaded, False otherwise (bool)
    """
    return model is not None

    
def load_model(model_path: str = "./models", force_download: bool = False):
    """
    Loads model from model_path
    model_path: path to model (str)
    force_download: force download model (bool)
    port_number: port number to load model (int)
    return: None
    """
    if check_model_loaded():
        return None
    if not model_path:
        raise ValueError("model_path is None")
    if (not os.path.exists(model_path)) or force_download:
        download_model(REPOSITORY, model_path, force_download = force_download)
    # load model
    global model
    print("Loading model")
    # precisions
    model = load_model_hf(model_path)
    return None

def predict_tags(prob_list:np.ndarray, threshold=0.5, model_path:str="./") -> List[str]:
    """
    Predicts tags from prob_list
    prob_list: list of probabilities, first 4 are ratings, rest are tags
    threshold: threshold for tags (float)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model, general_tags, character_tags
    probs = np.array(prob_list)
    #ratings = probs[:4] # first 4 are ratings
    #rating_index = np.argmax(ratings)
    tags = probs[4:] # rest are tags
    if general_tags is None or character_tags is None:
        read_tags(model_path)
    assert general_tags is not None and character_tags is not None, "general_tags and character_tags are not loaded"
    result = []
    for i, p in enumerate(tags):
        if i < len(general_tags) and p > threshold:
            tag_name = general_tags[i]
            # replace _ with space
            tag_name = tag_name.replace("_", " ")
            result.append(tag_name)
    return result

def predict_image(image: np.ndarray, model_path: str = "./") -> List[str]:
    """
    Predicts image from image
    image: image to predict (np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    probs = model.predict(image)[0]
    return predict_tags(probs, model_path=model_path)

def predict_images_batch(images: Generator, model_path: str = "./", batch_size = 16, minibatch_size = 16,total:int=-1, action:callable = None, threadexecutor:ThreadPoolExecutor = None) -> List[List[str]]:
    """
    Predicts images from images
    images: images to predict (list of np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    assert images is not None, "images is None"
    #images = [preprocess_image(image) for image in images]
    results = []
    for i in tqdm(range(max(1,total// batch_size)), desc="GPU Batch"):
        batch = []
        paths = []
        for j in tqdm(range(batch_size), desc=f"Loading batch {i}"):
            try:
                path, image = next(images)
                assert isinstance(image, Image.Image), f"Expected image to be Image.Image, got {type(image)} with value {image}"
                assert isinstance(path, str), f"Expected path to be str, got {type(path)} with value {path}"
                batch.append(image)
                paths.append(path)
            except StopIteration:
                break
        #batch = images[i*batch_size:(i+1)*batch_size]
        batch_processed = []
        for image in tqdm(batch, desc="Preprocessing"):
            batch_processed.append(preprocess_image(image))
        batch = np.array(batch_processed)
        # With proper handling, the later part can be threaded (GPU side, load next batch while predicting)
        threadexecutor.submit(threaded_job, batch, paths, minibatch_size, model_path, action)
        #threaded_job(batch, paths, minibatch_size, model_path, action)
    return results

def threaded_job(batch, paths, minibatch_size, model_path, action):
    try:
        probs = model.predict(batch, batch_size=minibatch_size)
        # move to cpu (tensorflow-gpu)
        # clear session
        keras.backend.clear_session()
        tags_batch = []
        for prob in probs:
            tags = predict_tags(prob, model_path=model_path)
            tags_batch.append(tags)
        del probs
        if action is not None:
            action(paths, tags_batch)
    except Exception as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            raise e
        return None

def handle_yield(image_path_list:List[str]):
    # yields image_path, image if image_path is valid and image loads
    for image_path in image_path_list:
        try:
            image = Image.open(image_path)
            yield image_path, image
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            continue

# for image paths, locally
def predict_images_from_path(image_paths: List[str], model_path: str = "./", action=None, batch_size=2048,minibatch_size=16) -> List[List[str]]:
    """
    Predicts images from image_paths
    image_paths: paths to images (list of str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # check if model is loaded
    if not check_model_loaded():
        global model
        if model is None:
            load_model(os.path.abspath(model_path))
    generator = handle_yield(image_paths)
    executor = ThreadPoolExecutor(max_workers=1)
    # batch size is 2048, minibatch size is 16 - batch size (RAM) / minibatch size (GPU)
    return predict_images_batch(generator, model_path=model_path, action=action, total=len(image_paths), batch_size=batch_size,minibatch_size = minibatch_size, threadexecutor=executor)

# using glob, get all images in a folder then request
def predict_local_path(path:str, recursive:bool=False, action:callable=None, max_items:int=0, batch_size=2048, minibatch_size=16) -> dict[str, List[str]]: # path: path to folder
    """
    Predicts images from path
    path: path to folder (str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # get all images in path
    import glob
    paths = []
    if not recursive:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            paths.extend(glob.glob(os.path.join(path, ext)))
    else:
        #os.walk
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".webp"):
                    paths.append(os.path.join(root, file))
    if max_items > 0:
        paths = paths[:max_items]
    print(f"Found {len(paths)} images")
    # post and get
    result = predict_images_from_path(paths, action=action, batch_size=batch_size, minibatch_size=minibatch_size)
    result_dict = {x[0]:x[1] for x in zip(paths, result)}
    return result_dict

def move_matching_items(paths:List[str], tags:List[List[str]]):
    """
    Moves images to matching tags
    """
    #print(f"Given tags: {tags[:10]}")
    #lambda x: any("futa" in y for y in x) -> move to D:\interrogate\matches
    for path, tag in tqdm(zip(paths, tags), desc="Moving matching items"):
        if any("futa" in y for y in tag):
            # move to D:\interrogate\matches
            print(f"Moving {path} to D:\\interrogate\\matches")
            target_path = os.path.join(r"D:\interrogate\matches", os.path.basename(path))
            if os.path.exists(os.path.join(r"D:\interrogate\matches", os.path.basename(path))):
                _i = 0
                while os.path.exists(target_path):
                    target_path = os.path.join(r"D:\interrogate\matches", f"{os.path.basename(path)}_{_i}")
                    _i += 1
            os.rename(path, target_path)
            print(f"Moved {path} to {target_path}")
# adjust by yourself
predict_local_path(r"D:\naidataset-1224", recursive=True, action=move_matching_items, max_items = 0,batch_size=256, minibatch_size=32).values() # ban tags

