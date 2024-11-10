from PIL import Image
import random
import torch
import time
from logger import create_logger

# Ensure the scaler is set properly for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=True)
scaler_disc = torch.cuda.amp.GradScaler(enabled=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

params_to_update = [param for param in model.parameters() if param.requires_grad]
vq_loss = VQLoss(
    disc_start=20000,
    disc_weight=0.5,
    disc_type='patchgan',
    disc_loss='hinge',
    gen_adv_loss='hinge',
    image_size=256,
    perceptual_weight=1.0,
    reconstruction_weight=1.0,
    reconstruction_loss='l2',
    codebook_weight=1.0,
).to(device)

optimizer = torch.optim.Adam(params_to_update, lr=1e-4, betas=(0.9, 0.95))
optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.95))
checkpoint_dir = 'checkpoints'
experiment_dir = 'experiments'
running_loss = 0
log_steps = 0
start_time = time.time()
model.train()
vq_loss.train()
train_steps = 0
logger = create_logger(os.getcwd(), 'train.log')
logger.info(f"Experiment directory created at {experiment_dir}")
for epoch in range(100):
    for i, (high_res_image, low_res_image) in enumerate(loader):
        high_res_image = high_res_image.to(device)
        low_res_image = low_res_image.to(device)

        # Zero gradients
        optimizer.zero_grad()
        optimizer_disc.zero_grad()

        # Forward pass
        recons_imgs, codebook_loss = model(high_res_image)
        loss_gen = vq_loss(codebook_loss, low_res_image, recons_imgs, optimizer_idx=0, global_step=train_steps+1, last_layer=model.decoder.last_layer,logger=logger, log_every=args.log_every)
        
        # Backward pass for generator
        scaler.scale(loss_gen).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params_to_update, 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Backward pass for discriminator
        loss_disc = vq_loss(codebook_loss, low_res_image, recons_imgs, optimizer_idx=1, global_step=train_steps+1,logger=logger, log_every=args.log_every)
        scaler_disc.scale(loss_disc).backward()
        scaler_disc.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad_norm_(vq_loss.discriminator.parameters(), 1.0)
        scaler_disc.step(optimizer_disc)
        scaler_disc.update()

        # Log loss values:
        running_loss += loss_gen.item() + loss_disc.item()
        log_steps += 1
        train_steps += 1

        if train_steps % 100 == 0:
            # Measure training speed:
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)
            avg_loss = running_loss / log_steps
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            running_loss = 0
            log_steps = 0
            start_time = time.time()

        # Save checkpoint:
        if train_steps % 5000 == 0 and train_steps > 0:
            model_weight = model.state_dict()
            checkpoint = {
                "model": model_weight,
                "optimizer": optimizer.state_dict(),
                "discriminator": vq_loss.discriminator.state_dict(),
                "optimizer_disc": optimizer_disc.state_dict(),
                "steps": train_steps,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
