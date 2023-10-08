import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("results.csv")
print(results)
print(results.dtypes)

noopt = results[results["extrinsics optimised"] == " False"]
opt = results[results["extrinsics optimised"] == " True"]

noopt = noopt[noopt["rot_noise"] == 0]
opt = opt[opt["rot_noise"] == 0]

noopt = noopt.groupby("trans_noise").mean()
opt = opt.groupby("trans_noise").mean()


print(noopt)
print(opt)

noopt["psnr"].plot(label="Noisy")
opt["psnr"].plot(label="Optimised")
plt.xlabel("Translation noise sigma (m)")
plt.ylabel("PSNR (dB)")
plt.title('Comparing PSNR for optimised and non-optimised extrinsics')
plt.legend()

plt.figure()
noopt["ssim"].plot()
opt["ssim"].plot()
plt.xlabel("Translation noise sigma (m)")
plt.ylabel("SSIM")
plt.title('Comparing SSIM for optimised and non-optimised extrinsics')
plt.legend()
plt.show()

