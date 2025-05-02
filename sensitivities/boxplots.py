import pandas as pd
import matplotlib.pyplot as plt

# Load each CSV
df1 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/sensitivity/FY/sensitivity_FY_i1_s1.csv')
df2 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/sensitivity/FY/sensitivity_FY_i1_s2.csv')
df3 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/sensitivity/FY/sensitivity_FY_i1_s3.csv')
df4 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/sensitivity/FY/sensitivity_FY_i1_s4.csv')
df5 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/sensitivity/FY/sensitivity_FY_i1_s5.csv')

# Prepare data
data = [
    df1['tbh'].rename('1'),
    df2['tbh'].rename('2'),
    df3['tbh'].rename('3'),
    df4['tbh'].rename('4'),
    df5['tbh'].rename('5'),
]
combined = pd.concat(data, axis=1)

# Custom flier properties 
flier_props = dict(marker='.',
                   markerfacecolor='none', 
                   markeredgecolor='black', 
                   markersize=10)

# Plot boxplot without visible outliers
plt.figure(figsize=(16, 6))
combined.boxplot(flierprops=flier_props)
plt.xlabel('Number of snow layers')
plt.ylabel('Horizontal brightness temperature [K]')
plt.ylim(110, None)  #
plt.grid(True)
plt.tight_layout()
plt.show()
