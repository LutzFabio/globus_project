import pandas as pd
from PIL import Image
import numpy as np
import os
import sys

def main(base_path):

    df_with_patches=pd.DataFrame(columns=['name','mean_alpha','std_alpha','all255'])

    for path, subdirs, files in os.walk(base_path):
        for name in files:
            if name.endswith('.png'):
                fetures=[]
                img = Image.open(os.path.join(path, name))
                alpha = np.array(img.split()[-1])
                fetures.append(name[:-4])
                fetures.append(np.mean(alpha))
                fetures.append(np.std(alpha))
                fetures.append(1) if np.all(alpha==255) else fetures.append(0)

                df_with_patches=df_with_patches.append({
                    'name':fetures[0], 'mean_alpha':fetures[1],
                    'std_alpha':fetures[2], 'all255':fetures[3],
                }, ignore_index=True)

    df_with_patches.to_csv(base_path+'data_on_alpha.csv',index=False)

if __name__ == '__main__':

    if len(sys.argv)>1:
        base_path=sys.argv[1]
    else:
        base_path = '/Volumes/DS_immersive/DS201909/Globus/mashas_sandbox/'
    main(base_path)
