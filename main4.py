from image_analysis import compute_tb_from_rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap,dtu_grey
from dictionaries import CIMR
# -------------------------- User Input -------------------------- # 

# Max and min from dics
tb_max = CIMR["FYI"]["36.5H"]["statistic"]["max"]
tb_min = CIMR["FYI"]["36.5H"]["statistic"]["min"]

# Convolved image
convolved_img_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/convolved_tb.png"
output_convolved_path = r"C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/out"

# Resampled image
resampled_img_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/output.png"
output_resampled_path = r"C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/out2"

#%% ---------------------------------------------------------------- # 

convolved_img_tb = compute_tb_from_rgb(convolved_img_path, tb_min, tb_max, output_convolved_path)
resampled_img_tb = compute_tb_from_rgb(resampled_img_path, tb_min, tb_max, output_resampled_path)
#%%
tb_diff = resampled_img_tb - convolved_img_tb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create regular image plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
im = ax.imshow(tb_diff, cmap=dtu_coolwarm_cmap, vmin=-5, vmax=5, origin='upper')
ax.axis('off')


# Add colorbar to main axes
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
cbar.set_label("TB Difference (K)")

# Save and show
plt.savefig("tb_difference_with_land_overlay.png", bbox_inches='tight', pad_inches=0)
plt.show()




