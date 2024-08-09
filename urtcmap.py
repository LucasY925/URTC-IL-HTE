# MAP distribution histogram
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')['s_mapritread2122f'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of MAP Reading Scores (s_mapritread2122f)', fontsize=15)
plt.xlabel('MAP Reading Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)

plt.show()
