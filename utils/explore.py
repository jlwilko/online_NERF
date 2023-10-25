#%%
import pandas as pd
import json
import os

print(os.getcwd())

with open("../data/lfodo/core/seq24/incremental/inc_temporal_results.json") as f:
	data = json.load(f)


# %% 
import matplotlib.pyplot as plt

# df = pd.DataFrame(data)
df = pd.json_normalize(data)
print(df)
print(type(df))
print(df.dtypes)

print(df["test_set_stats"])

for idx in range(len(df)):
	run_stats = pd.json_normalize(df["test_set_stats"][idx])
	plt.figure()
	# run_stats["psnr"].plot(label="PSNR")
	run_stats["lpips"].plot(label="LPIPS")
	plt.xlabel("Iteration")
	plt.ylabel("Metric")

	# set axes range 
	plt.ylim([0, 1])
	plt.xlim([0, 20])

	plt.title(f'PSNR for run {idx}')
	plt.legend()
	plt.savefig(f"run_{idx}_stats.png")
	plt.show()



	print(run_stats)


# for i in range(len(df)):
# 	# plot the test_set_stats for each iteration
# 	print(df["test_set_stats"][i])
# 	print(df["test_set_stats"][i]["psnr"])
# 	print(df["test_set_stats"][i]["ssim"])
# 	print(df["test_set_stats"][i]["lpips"])

# %%
plt.figure()
df["training_time"].plot()


#%% 
last_stats = data[-1]["test_set_stats"]
last_stats = pd.json_normalize(last_stats)
print(f"psnr = {last_stats['psnr'].mean()}")
print(f"ssim = {last_stats['ssim'].mean()}")
print(f"lpips = {last_stats['lpips'].mean()}")


print(f"training size = {data[-1]['training_set_size']}")
print(f"training time = {data[-1]['training_time']}")