import maximize_AUROCFIT
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
n_fold = 3 # 3 * 10 만큰 fold 함
n_est = 20 # leaf 갯수
xlsx = [500,1000,2000,5000,10000]
#for num in xlsx:
#    Oncetrain.oncefit(num,n_fold,1)
epochs = 10 #epoch -> 십만으로

maximize_AUROCFIT.oncefit(500,n_fold,epochs)

maximize_AUROCFIT.oncefit(1000,n_fold,epochs)

maximize_AUROCFIT.oncefit(2000,n_fold,epochs)

maximize_AUROCFIT.oncefit(5000,n_fold,epochs)

maximize_AUROCFIT.oncefit(10000,n_fold,epochs)
