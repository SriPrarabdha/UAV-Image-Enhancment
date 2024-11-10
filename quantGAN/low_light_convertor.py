import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import numpy as np
from PIL import Image

from vq_model import VQ_models
import os
from custom_dataset import HighLowResDataset_test, HighLowResDataset_normal

# from vq_model import VQ_models

def find_correct_folders(scan_dir):
    correct_folders = []

    with os.scandir(scan_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                skybox_path = os.path.join(entry.path, 'matterport_skybox_images')
                if os.path.exists(skybox_path) and os.path.isdir(skybox_path):
                    correct_folders.append(skybox_path)
    return correct_folders

def arrange_images(skybox_path):
    sami_images = []
    small_images = []


    with os.scandir(skybox_path) as entries:
        for entry in entries:
            if entry.is_file():
                split = entry.name.split('_')
                if split[-1] == 'sami.jpg':
                    sami_images.append(entry.path)
                elif split[-1] == 'small.jpg' and split[-2] != 'depth':
                    small_images.append(entry.path)
        sami_images = sorted(sami_images)    
        small_images = sorted(small_images)
    return sami_images, small_images

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    model.load_state_dict(model_weight)
    del checkpoint

    # dataloader and dataset
    folders_to_iterate = find_correct_folders(args.input_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # for folder in folders_to_iterate:
    # sami_images, small_images = arrange_images(folder)
    ip_dir = "/media/mlr_lab/325C37DE7879ABF2/prarabda/amazon_challenge/cv/normal_light_images"
    imgs = os.listdir(ip_dir)
    imgs = [ip_dir + '/' + img for img in imgs]
    print(imgs)
    dataset = HighLowResDataset_normal(imgs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # output_dir = os.path.join(args.output_dir, folder.split('/')[-2])
    output_dir = "/media/mlr_lab/325C37DE7879ABF2/prarabda/amazon_challenge/cv/gan_low_light_images"
    # imgs = os.listdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ## check the output dir being created
    for idx, batch in enumerate(dataloader):
        # output dir
        # batch = torch.squeeze(batch) 
        batch = batch.to(device)   
        
        # out_path = imgs[idx].replace('.jpg', '_{}.jpg'.format(args.suffix))
        # out_path = out_path.replace('.jpeg', '_{}.jpeg'.format(args.suffix))
        # out_path = out_path.replace('.png', '_{}.png'.format(args.suffix))
        # out_filename = out_path.split('/')[-1]
        print(imgs[idx].split('/')[-1])
        out_path = os.path.join(output_dir, imgs[idx].split('/')[-1])
        with torch.no_grad():
            latent, _, [_, _, indices] = model.encode(batch)
            output = model.decode_code(indices, latent.shape)
        
        output = output.permute(0, 2, 3, 1)
        output = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        output_image = Image.fromarray(output[0])
#             output_image = Image.new('RGB', (1024, 1024))

# # Pasting the images in the order described
#             output_image.paste(Image.fromarray(output[0]), (0, 0))
#             output_image.paste(Image.fromarray(output[1]), (256, 0))
#             output_image.paste(Image.fromarray(output[2]), (512, 0))
#             output_image.paste(Image.fromarray(output[3]), (768, 0))

#             output_image.paste(Image.fromarray(output[4]), (0, 256))
#             output_image.paste(Image.fromarray(output[5]), (256, 256))
#             output_image.paste(Image.fromarray(output[6]), (512, 256))
#             output_image.paste(Image.fromarray(output[7]), (768, 256))

#             output_image.paste(Image.fromarray(output[8]), (0, 512))
#             output_image.paste(Image.fromarray(output[9]), (256, 512))
#             output_image.paste(Image.fromarray(output[10]), (512, 512))
#             output_image.paste(Image.fromarray(output[11]), (768, 512))

#             output_image.paste(Image.fromarray(output[12]), (0, 768))
#             output_image.paste(Image.fromarray(output[13]), (256, 768))
#             output_image.paste(Image.fromarray(output[14]), (512, 768))
#             output_image.paste(Image.fromarray(output[15]), (768, 768))

        # Save or show the final image
        output_image.save(out_path)
        # output_image.show()
        print("Reconstructed image is saved to {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/media/mlr_lab/325C37DE7879ABF2/prarabda/amazon_challenge/cv/normal_light_images")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--suffix", type=str, default="tokenizer_image")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str, default="/media/mlr_lab/325C37DE7879ABF2/LowLIGHTQuantGAN/0015000.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512, 1024], default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)


# if __name__ == "__main__":
#     arrange_images('/media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/Matterport3DSimulator/data/v1/scans/PX4nDJXEHrG/matterport_skybox_images')