import pickle
import random

from torchvision import transforms

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def image_to_tensor(image):
    return _IMAGE_TRANSFORM(image)


def load_pair_index(path):
    """Loads the pre-computed training pair index of one sequence.

    The file stores two lists indexed by frame: candidate second frames within the
    translation range, and candidate second frames with significant rotation.
    """
    with open(path, 'rb') as f:
        pair_index = pickle.load(f)
    return pair_index[0], pair_index[1]


def sample_second_frame(nearby, nearby_rot, rot_threshold):
    """Samples the second frame of a training pair, preferring rotation pairs.

    Returns None if the first frame has no valid candidate.
    """
    if len(nearby_rot) != 0 and len(nearby) != 0:
        if random.randint(1, 10) <= rot_threshold:
            return random.choice(nearby_rot)
        return random.choice(nearby)
    if len(nearby_rot) != 0:
        return random.choice(nearby_rot)
    if len(nearby) != 0:
        return random.choice(nearby)
    return None


def sample_training_pair(idx, nearby_points, nearby_points_rot, num_items, rot_threshold):
    """Returns (idx1, idx2) of a training pair starting from frame ``idx``.

    Frames without any valid candidate are replaced by a randomly drawn frame.
    """
    idx1 = idx
    idx2 = sample_second_frame(nearby_points[idx1], nearby_points_rot[idx1], rot_threshold)
    while idx2 is None:
        idx1 = random.randint(0, num_items - 1)
        idx2 = sample_second_frame(nearby_points[idx1], nearby_points_rot[idx1], rot_threshold)
    return idx1, idx2
