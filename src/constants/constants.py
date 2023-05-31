BASE_DIR_PATH = "/srv/data/dataset_for_bacteria_generation"
WORK_DIR_PATH = "/usr/src"

# ========================== initial data ==========================

SEGMENTATION_OBJECT_PATH = f"{BASE_DIR_PATH}/SegmentationObject"
SEGMENTATION_CLASS_PATH = f"{BASE_DIR_PATH}/SegmentationClass"
ORIGINAL_IMAGES_PATH = f"{BASE_DIR_PATH}/OriginalImages"

# ========================== data ==========================

_BACTERIAS = "bacterias"
_SUBSTRATES = "substrates"
_MASKS = "masks"

_DATA = "data"
_FAKE = "fake"
_TEST = "test"
_TRAIN = "train"
_ORIGINAL = "original"
_VALIDATION = "validation"

TRAIN_FAKE_BACTERIAS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_FAKE}/{_TRAIN}/{_BACTERIAS}"
TRAIN_FAKE_MASKS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_FAKE}/{_TRAIN}/{_MASKS}"
TEST_FAKE_BACTERIAS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_FAKE}/{_TEST}/{_BACTERIAS}"
TEST_FAKE_MASKS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_FAKE}/{_TEST}/{_MASKS}"

ORIGINAL_BACTERIAS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_ORIGINAL}/{_BACTERIAS}"
ORIGINAL_SUBSTRATES_64x64_PATH = (
    f"{WORK_DIR_PATH}/{_DATA}/{_ORIGINAL}/{_SUBSTRATES}_64x64"
)
ORIGINAL_SUBSTRATES_128x128_PATH = (
    f"{WORK_DIR_PATH}/{_DATA}/{_ORIGINAL}/{_SUBSTRATES}_128x128"
)

VALIDATION_BACTERIAS_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_VALIDATION}/{_BACTERIAS}"
VALIDATION_SUBSTRATES_PATH = f"{WORK_DIR_PATH}/{_DATA}/{_VALIDATION}/{_SUBSTRATES}"

# ========================== models ==========================
