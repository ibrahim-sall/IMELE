import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import glob
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
import argparse
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import csv
import re
import logging

"""_summary_

python test.py --model '/data/Block0_skip_model_110.pth.tar' --csv './dataset/test.csv' --outfile '/data/out.tiff'

Returns:
    _type_: _description_
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the main function.")
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--csv")
    parser.add_argument("--out")
    args = parser.parse_args()

    logging.info(f"Arguments received: model={args.model}, csv={args.csv}, outfile={args.out}")

    md = glob.glob(args.model)
    md.sort(key=natural_keys)
    logging.info(f"Found {len(md)} model checkpoints.")

    writer = SummaryWriter(log_dir="./logs")

    for x in md:
        logging.info(f"Processing model checkpoint: {x}")
        x = str(x)

        model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
        
        # Load the model checkpoint onto the CPU
        state_dict = torch.load(x, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict, strict=False)
        logging.info("Model state_dict loaded successfully.")

        test_loader = loaddata.getTestingData(1, args.csv)
        logging.info("Testing data loader initialized.")

        test(test_loader, model, args, writer)

    writer.close()
    logging.info("Finished processing all model checkpoints.")


def test(test_loader, model, args, writer):
    logging.info("Starting the test function.")
    losses = AverageMeter()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'SSIM': 0}

    for i, sample_batched in enumerate(test_loader):
        logging.info(f"Processing batch {i + 1}.")
        image, depth = sample_batched['image'], sample_batched['depth']
        depth = depth.to(device, non_blocking=True)
        image = image.to(device)
        output = model(image)

        output = torch.nn.functional.interpolate(output, size=(440, 440), mode='bilinear')

        batchSize = depth.size(0)
        logging.info(f"Batch size: {batchSize}")

        testing_loss(depth, output, losses, batchSize)
        logging.info(f"Loss for batch {i + 1}: {losses.val:.4f}")

        totalNumber += batchSize

        errors = util.evaluateError(output, depth, i, batchSize)
        logging.info(f"Errors for batch {i + 1}: {errors}")

        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)
        logging.info(f"Average errors after batch {i + 1}: {averageError}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Batch", losses.avg, i)
        writer.add_scalar("Metrics/MSE", averageError['MSE'], i)
        writer.add_scalar("Metrics/RMSE", np.sqrt(averageError['MSE']), i)
        writer.add_scalar("Metrics/MAE", averageError['MAE'], i)
        writer.add_scalar("Metrics/SSIM", averageError['SSIM'], i)

        output_np = output.squeeze().cpu().detach().numpy() 
        # output_file = os.path.join(args.out, f"output_batch_{i + 1}.tif")
        # logging.info(f"Saving output to {output_file}")
        # output_image = Image.fromarray(output_np.astype(np.uint16)) 
        # output_image.save(output_file)
        
        logging.info(f"Output saved successfully.")
        
    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    loss = float((losses.avg).data.cpu().numpy())

    logging.info(f"Final metrics: Loss={loss:.4f}, MSE={averageError['MSE']:.4f}, "
                 f"RMSE={averageError['RMSE']:.4f}, MAE={averageError['MAE']:.4f}, "
                 f"SSIM={averageError['SSIM']:.4f}")

    print('Model Loss {loss:.4f}\t'
          'MSE {mse:.4f}\t'
          'RMSE {rmse:.4f}\t'
          'MAE {mae:.4f}\t'
          'SSIM {ssim:.4f}\t'.format(loss=loss, mse=averageError['MSE'],
                                      rmse=averageError['RMSE'], mae=averageError['MAE'],
                                      ssim=averageError['SSIM']))
    
def testing_loss(depth, output, losses, batchSize):
    logging.info("Calculating testing loss.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(device)
    get_gradient = sobel.Sobel().to(device)
    cos = nn.CosineSimilarity(dim=1, eps=0)
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    loss = loss_depth + loss_normal + (loss_dx + loss_dy)
    losses.update(loss.data, batchSize)
    logging.info(f"Loss components: depth={loss_depth:.4f}, dx={loss_dx:.4f}, dy={loss_dy:.4f}, normal={loss_normal:.4f}, total={loss:.4f}")


def define_model(is_resnet, is_densenet, is_senet):
    logging.info(f"Defining model: is_resnet={is_resnet}, is_densenet={is_densenet}, is_senet={is_senet}")
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    logging.info("Model defined successfully.")
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    main()
