import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_stats_rot_noise(df):
	df = df.drop(columns=["datetime"])
	trans_df = df[df["trans_noise"] == 0]
	noopt = trans_df[trans_df["extrinsics_optimised"] == False]
	opt = trans_df[trans_df["extrinsics_optimised"] == True]

	noopt = noopt.groupby("rot_noise").mean()
	opt = opt.groupby("rot_noise").mean()

	plt.rcParams.update({'font.size': 14})
	plt.figure()
	noopt["psnr"].plot(label="Optimised")
	opt["psnr"].plot(label="Noisy")
	plt.xlabel("Rotation noise sigma ($^\circ$)")
	plt.ylabel("PSNR (dB)")
	plt.title('PSNR with rotational noise')
	plt.grid()
	plt.xlim([0, 4])
	plt.ylim([15, 30])
	plt.legend()
	plt.savefig("../writing/img/psnr_rot_noise.eps", bbox_inches='tight')

	plt.rcParams.update({'font.size': 14})
	plt.figure()
	noopt["ssim"].plot(label="Optimised")
	opt["ssim"].plot(label="Noisy")
	plt.xlabel("Rotation noise sigma ($^\circ$)")
	plt.ylabel("SSIM")
	plt.title('SSIM with rotational noise')
	plt.grid()
	plt.xlim([0, 4])
	plt.ylim([0, 1])
	plt.legend()
	plt.savefig("../writing/img/ssim_rot_noise.eps", bbox_inches='tight')

	plt.rcParams.update({'font.size': 14})
	plt.figure()
	opt["lpips"].plot(label="Optimised")
	noopt["lpips"].plot(label="Noisy")
	plt.xlabel("Rotation noise sigma ($^\circ$)")
	plt.ylabel("LPIPS")
	plt.title('LPIPS with rotational noise')
	plt.grid()
	plt.xlim([0, 4])
	plt.ylim([0, 1])
	plt.legend()
	plt.savefig("../writing/img/lpips_rot_noise.eps", bbox_inches='tight')

def plot_stats_trans_noise(df):
	df = df.drop(columns=["datetime"])	
	trans_df = df[df["rot_noise"] == 0]
	noopt = trans_df[trans_df["extrinsics_optimised"] == False]
	opt = trans_df[trans_df["extrinsics_optimised"] == True]

	noopt = noopt.groupby("trans_noise").mean()
	opt = opt.groupby("trans_noise").mean()

	plt.rcParams.update({'font.size': 12})
	plt.figure()
	noopt["psnr"].plot(label="Optimised")
	opt["psnr"].plot(label="Noisy")
	plt.xlabel("Translation noise sigma (m)")
	plt.ylabel("PSNR (dB)")
	plt.title('PSNR with translational noise')
	plt.grid()
	plt.xlim([0, 0.4])
	plt.ylim([15, 30])
	plt.legend()
	plt.savefig("../writing/img/psnr_trans_noise.eps", bbox_inches='tight')

	plt.rcParams.update({'font.size': 14})
	plt.figure()
	noopt["ssim"].plot(label="Optimised")
	opt["ssim"].plot(label="Noisy")
	plt.xlabel("Translation noise sigma (m)")
	plt.ylabel("SSIM")
	plt.title('SSIM with translational noise')
	plt.grid()
	plt.xlim([0, 0.4])
	plt.ylim([0, 1])
	plt.legend()
	plt.savefig("../writing/img/ssim_trans_noise.eps", bbox_inches='tight')

	plt.rcParams.update({'font.size': 14})
	plt.figure()
	opt["lpips"].plot(label="Optimised")
	noopt["lpips"].plot(label="Noisy")
	plt.xlabel("Translation noise sigma (m)")
	plt.ylabel("LPIPS")
	plt.title('LPIPS with translational noise')
	plt.grid()
	plt.xlim([0, 0.4])
	plt.ylim([0, 1])
	plt.legend()
	plt.savefig("../writing/img/lpips_trans_noise.eps", bbox_inches='tight')

def plot_ate_results():
	x = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
	y = np.array([0.000, 0.0318, 0.0426,0.113,0.172,0.218])
	z = np.array([0.02,0.0196,0.0234,0.0268,0.0306,0.0545])

	plt.figure()
	# plot x v y
	plt.plot(x, y, label="Noisy")
	plt.plot(x, z, label="Optimised")
	plt.xlabel("Translation noise sigma (m)")
	plt.ylabel("ATE (m)")
	plt.title('ATE with translational noise')
	plt.grid()

	# plt.savefig("../writing/img/ate.eps", bbox_inches='tight')




	return


if __name__ == "__main__":
	print("Plotting results...")

	plt.rcParams.update({
		"text.usetex": True,
	})
	dfs = []
	for filename in glob.glob("data/lfodo/core/seq*/results.csv"):
		dfs.append(pd.read_csv(filename))
	df = pd.concat(dfs)
	print(df.columns)

	# plot_stats_trans_noise(df)
	# plot_stats_rot_noise(df)
	plot_ate_results()
	plt.show()

	# df = pd.read_csv("results.csv")
