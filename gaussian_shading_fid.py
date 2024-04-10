import argparse
import copy
from tqdm import tqdm
import json
import torch
import os
from statistics import mean, stdev
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from image_utils import *
from pytorch_fid.fid_score import *


def main(args):
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # dataset
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_files = dataset['images']
        dataset = dataset['annotations']
        prompt_key = 'caption'

    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    w_dir = f'./fid_outputs/coco/{args.run_name}/w_gen'
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    for i in tqdm(range(0, args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w= outputs.images[0]
        image_file_name = image_files[i]['file_name']
        image_w.save(f'{w_dir}/{image_file_name}')

    #calculate fid
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    fid_value_w = calculate_fid_given_paths([args.gt_folder, w_dir],
                                            50,
                                            device,
                                            2048,
                                            num_workers)

    with open(args.output_path + 'fid.txt', "a") as file:
        file.write('model:' + args.model_path + '       '+ 'fid_w:' + str(fid_value_w) + '\n')
    print(f'fid_w: {fid_value_w}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--run_name', default='Gaussian_Shading')
    parser.add_argument('--num', default=5000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--prompt_file', default='./fid_outputs/coco/meta_data.json')
    parser.add_argument('--gt_folder', default='./fid_outputs/coco/ground_truth')
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')

    args = parser.parse_args()

    main(args)
