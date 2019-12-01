import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[.88,.12],
     [.3,.40]]        
df_cm = pd.DataFrame(array, range(2),
                  [1,0])

sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


plt.show()
