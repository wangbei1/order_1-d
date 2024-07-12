from easydict import EasyDict as edict

DefaultDataPath = edict()

DefaultDataPath.ImageNet = edict()
DefaultDataPath.FFHQ = edict()
DefaultDataPath.CelebAHQ = edict()
# DefaultDataPath.ImageNet.root = "Your Data Path/Datasets/ImageNet"
# DefaultDataPath.ImageNet.train_write_root = "Your Data Path/Datasets/ImageNet/train"
# DefaultDataPath.ImageNet.val_write_root = "Your Data Path/Datasets/ImageNet/val"

DefaultDataPath.ImageNet.root = "/private/task/linyijing/dataset/imagenet"
DefaultDataPath.ImageNet.train_write_root = "/private/task/linyijing/dataset/imagenet/train"
DefaultDataPath.ImageNet.val_write_root = "/private/task/linyijing/dataset/imagenet/val"

DefaultDataPath.FFHQ.root = "/private/task/jwn/Dataset/FFHQ/jiawn/Datasets"
DefaultDataPath.FFHQ.train_lmdb = "/private/task/jwn/Dataset/FFHQ/jiawn/Datasets"
DefaultDataPath.FFHQ.val_lmdb = "/private/task/jwn/Dataset/FFHQ/jiawn/Datasets/FFHQ"

DefaultDataPath.CelebAHQ.root = "/private/task/jwn/Dataset/CelebA-HQ"