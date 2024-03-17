import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
import argparse
from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)

def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/mnt/Datasets/Restoration')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=268, help='scale images to this size') #572,268
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

train_batch_size = 10
num_samples = 1
sum_scale = 0.01
image_size = 256
condition = True
opt = parsr_args()

results_folder = "./ckpt_universal/diffuir"

if 'universal' in results_folder:
    dataset_fog = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='fog')
    dataset_light = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='light_only')
    dataset_rain = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='rain')
    dataset_snow = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='snow')
    dataset_blur = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='blur')
    dataset = [dataset_fog, dataset_light, dataset_rain, dataset_snow, dataset_blur]
    
    num_unet = 1
    objective = 'pred_res'
    test_res_or_noise = "res"
    train_num_steps = 300000
    train_batch_size = 10
    sum_scale = 0.01
    delta_end = 1.8e-3


model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise
)
diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,           # number of steps
    delta_end = delta_end,
    sampling_timesteps=sampling_timesteps,
    objective=objective,
    loss_type='l1',            # L1 or L2
    condition=condition,
    sum_scale=sum_scale,
    test_res_or_noise = test_res_or_noise,
)

trainer = Trainer(
    diffusion,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    num_unet=num_unet,
)

# train
# trainer.load(30)
trainer.train()
