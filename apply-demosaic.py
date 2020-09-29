import argparse
import os
import torch
import numpy as np
from data import ProDemosaicDataset, SharpDemosaicDataset, N2SDataset
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
	parser.add_argument("--noconvert", nargs="?", type=bool, default=False, const=True)
	parser.add_argument("--dataset", type=str, default="pro", help="Type of dataset, sharp or pro")
	parser.add_argument("--device", type=str, default="cpu", help="The device to use for inference")
	parser.add_argument("--channelswap", type=bool, nargs="?", default=False, const=True, help="Use for older models that were trained on the dataset that swaps channels")
	parser.add_argument("--channels", type=int, default=2, help="number of denoising channels" )
	parser.add_argument("--save_input", action="store_true")
	parser.add_argument("--save_gt", action="store_true")
	parser.add_argument("--chop", type=int, default=1)
	parser.add_argument("--n2s", action="store_true")

	args = parser.parse_args()

	channels = args.channels
 
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
			crop=False)
	elif args.dataset == "direct_sharp":
		dataset = N2SDataset(
			args.inputdir,
			crop=False,
			sharp=True,
			channels=channels,
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
			net = Unet(n_channel_in=channels, n_channel_out=channels, **model_params)
		net.load_state_dict(state)
	else:
		net = state

	net.to(args.device)
	torch.no_grad()
	net.eval()

	for i in range(len(dataset)):
		sharp, pro = dataset.get_full(i)
		sharp = sharp.to(args.device)
		
		if args.n2s:
			print("N2S evening")
			sharp = sharp[:, :-(sharp.shape[1] % 16), sharp.shape[2] % 16:]
			pro = pro[:, :-(pro.shape[1] %16), pro.shape[2] % 16:]
			print(sharp.shape)


		paths = dataset.paths_grouped[i]
		basename = os.path.basename(paths[0])[:-9]

		if channels == 2:
			assert not np.all(sharp[0].detach().numpy() == sharp[1].detach().numpy())
			
			if args.channelswap:
				sharp = torch.stack((sharp[1], sharp[0]))
		
		if pro is not None:
			pro = pro.to('cpu').detach().numpy()

		prediction = np.zeros_like(pro)
		if args.chop > 1:
			idx = [int(x) for x in np.linspace(0, sharp.shape[-2], args.chop)] # indexes
			assert idx[-1] == sharp.shape[-2]
			for begin, end in zip(idx[:-1], idx[1:]):
				prediction[:, begin:end, :] = net(sharp[:, begin:end, :].unsqueeze(0)).squeeze(0).detach().numpy()
		else:
			prediction = net(sharp.unsqueeze(0)).squeeze(0).detach().numpy()
		
		if channels == 2:
			if args.channelswap:
				prediction = np.stack((prediction[1], prediction[0]), axis=0)

			prediction_high = Image.fromarray(((1.0 - prediction[0]) * 65535).astype(np.uint32))
			prediction_low = Image.fromarray(((1.0 - prediction[1]) * 65535).astype(np.uint32))

			prediction_high.save(os.path.join(args.outdir, basename+"_high.png"))
			prediction_low.save(os.path.join(args.outdir, basename+"_low.png"))
		else:
			prediction_high = Image.fromarray(((1.0 - prediction[0]) * 65535).astype(np.uint32))
			prediction_high.save(os.path.join(args.outdir, basename+"_high.png"))

		prediction_high.save(os.path.join(args.outdir, basename+"_high.png"))
		prediction_low.save(os.path.join(args.outdir, basename+"_low.png"))

		if args.save_gt:
			if pro is None:
				print("There is not GT to save for sharp data")
			gt_high = Image.fromarray(((1.0 - pro[0]) * 65535).astype(np.uint32))
			gt_low = Image.fromarray(((1.0 - pro[1]) * 65535).astype(np.uint32))
			gt_high.save(os.path.join(args.outdir, "gt_"+basename+"_high.png"))
			gt_low.save(os.path.join(args.outdir, "gt_"+basename+"_low.png"))

		if args.save_input:
			raise NotImplementedError

	if not args.noconvert:
		print("generated images, running high low conversion")
		os.system("yes | ./hilo_converter_v1.2 {} {}".format(args.outdir, args.outdir))

