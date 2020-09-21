import argparse
import os
import torch
import numpy as np
from data import ProDemosaicDataset
import matplotlib.pyplot as plt
from PIL import Image
import json
import model

if __name__=="__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("model")
	parser.add_argument("inputdir")
	parser.add_argument("outdir")
	parser.add_argument("--config", type=str, nargs="?", default="", help="Use the configs the model was trained with to load it")
	parser.add_argument("--class", type=str, nargs="?", default="unet", help="which model class to use")
	parser.add_argument("--statedict", type=str, nargs="?", const=True, default=False, help="whether the model save is a state dict. Assumes pickle elsewise.")
	parser.add_argument("--convert", type=bool, default=True)
	parser.add_argument("--dataset", type=str, default="pro", help="Type of dataset, sharp or pro")
	parser.add_argument("--device", type=str, default="cpu", help="The device to use for inference")
	parser.add_argument("--channelswap", type=bool, nargs="?", default=False, const=True, help="Use for older models that were trained on the dataset that swaps channels")

	args = parser.parse_args()
 
	if not os.path.exists(args.inputdir):
		raise ValueError("Input directory not found: {}".format(args.inputdir))

	if not os.path.exists(args.outdir):
		os.mkdir(args.outdir)

	config = {}
	if not args.config == "":
		with open(args.config, 'r') as f:
			config = json.load(f)

	if args.dataset == "pro":
		dataset = ProDemosaicDataset(
			args.inputdir,
			crop=False,
			patches_per_image=1)
	elif args.dataset == "sharp":
		dataset = SharpDemosaicDataset(
			args.inputdir,
			crop=False,
			patches_per_image=1)

	state = torch.load(args.model, map_location=args.device)
	if args.statedict:
		model_name = config.get("model", "unet")
		model_params = config.get("model_params", {})
		if  model_name == "resnet":
			net = model.ResNet(2, 2, **model_params)
		elif model_name == "unet":
			net = model.UNet(2, **model_params)
		elif model_name == "n2s-unet":
			from noise2self.models.unet import Unet
			net = Unet(n_channel_in=2, n_channel_out=2, **model_params)
		net.load_state_dict(state)
	else:
		net = state

	net.to(args.device)
	torch.no_grad()
	net.eval()

	for i in range(1):#range(len(dataset)):
		sharp, _ = dataset.get_full(i)
		paths = dataset.paths_grouped[i]
		basename = os.path.basename(paths[0])[:-9]
		assert not np.all(sharp[0].detach().numpy() == sharp[1].detach().numpy())
		
		if args.channelswap:
			sharp = torch.stack((sharp[1], sharp[0]))

		prediction = net(sharp.unsqueeze(0)).squeeze(0).detach().numpy()
		
		if args.channelswap:
			prediction = np.stack((prediction[1], prediction[0]), axis=0)

		prediction_high = ((1.0 - prediction[0]) * 65535).astype(np.uint32)
		prediction_low = ((1.0 - prediction[1]) * 65535).astype(np.uint32)

		im_high = Image.fromarray(prediction_high)
		im_low = Image.fromarray(prediction_low)

		im_high.save(os.path.join(args.outdir, basename+"_high.png"))
		im_low.save(os.path.join(args.outdir, basename+"_low.png"))

	if args.convert:
		print("generated images, running high low conversion")
		os.system("./hilo_converter_v1.2 {} {}".format(args.outdir, args.outdir))

