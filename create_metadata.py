import os
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import config

path = config.path

def load_data_from_folder(path):

   METADATA=[]
   for root, dirs, files in tqdm(os.walk(path, topdown=False)):
      for name in files:

         cheking = os.path.join(root, name).split("/")[-1][-4:]
         if cheking == '.wav':
            try :
               file_name  = os.path.join(root, name)
               dir_name=os.path.join(root, name).split("\\")[-2]

               METADATA.append([file_name,dir_name])

            except Exception as e:
               return
               pass

   df = pd.DataFrame(METADATA)
   # adding column name to the respective columns
   lb = LabelBinarizer()
   df.columns = ['relative_path', 'label']
   df['classID'] = df.loc[:, 'label'] #duplicate the label column

   lb.fit(df['classID'])
   df['classID'] = lb.transform(df['classID'])
   return df

