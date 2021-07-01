# Model 형성에 기여 하는 모듈들
import pandas as pd
import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _check_sample_weight
from sklearn.model_selection import train_test_split
# import 해야하는 모듈들
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import collections
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pyperclip
import math
from functools import reduce
import operator
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from scipy.stats import gmean
arr = []
for i in range(30):
    arr.append('예측0')
    arr.append('예측1')

conma=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])

warnings.filterwarnings('ignore')

"""
predict(estm_weights, gunT, budoT)
estm_weights와 건전 부도 데이터 가지고 예측한 값과 실제값을 리턴해줌

predict_proba(estm_weights, gunT, budoT)
estm_weights와 건전 부도 데이터 가지고 proba 리턴

geoacc(mat)
makeconma(conma,mat)

Find_Optimal_Cutoff(target, predicted)
실제값과 예측값을 가지고 적절한 Threshold return

get_proba(estm_weights,gunT,budoT)
geoacc return

get_roc_auc_score(estm_weights,gunT,budoT)
auroc score return



"""

def predict(estm_weights, gunT, budoT):
    Threshold = Find_Threshold(estm_weights,gunT,budoT)
    total = gunT.shape[0]+budoT.shape[0]
    proba = np.zeros((total,1))
    proba = np.array(proba)

    pred = np.zeros((total,1))
    pred = np.array(pred)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]
    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]

    for i in range(budoT.shape[0]):
        if proba[i+gunT.shape[0]] <= Threshold:
            pred[i+gunT.shape[0]] = 0
        else:
            pred[i+gunT.shape[0]] = 1
    for i in range(gunT.shape[0]):
        if proba[i] >= Threshold:
            pred[i] = 1

    return pred,real

def predict_proba(estm_weights,gunT,budoT):
    Threshold = Find_Threshold(estm_weights,gunT,budoT)
    #Threshold = 0.5
    total = gunT.shape[0]+budoT.shape[0]
    proba = np.zeros((total,1))
    proba = np.array(proba)
    pred = np.zeros((total,1))
    pred = np.array(pred)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    return proba

def geoacc(mat):
    value = (mat.iloc[0,0] / sum(mat.iloc[0,:])) * (mat.iloc[1,1] / sum(mat.iloc[1,:]))
    return math.sqrt(value)

def makeconma(conma,mat):
    conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    return conma

#-----------------roopre----------------
def Find_Optimal_Cutoff(target, predicted):

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index = i), 'threshold' : pd.Series(threshold, index = i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def Find_Threshold(estm_weights,gunT,budoT):
    gunT=np.array(gunT)
    budoT=np.array(budoT)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    proba = np.zeros((gunT.shape[0]+budoT.shape[0],1))
    proba = np.array(proba)
    for i in range(proba.shape[0]):
        proba[0] = 0

    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    return Find_Optimal_Cutoff(real,proba)

def normalize_weight(weight):
    total = 0
    for w in weight:
        total+=w

    for w in range(weight.shape[0]):
        weight[w] /= total
    return weight

def get_proba(estm_weights,gunT,budoT):
    Threshold = Find_Threshold(estm_weights,gunT,budoT)
    #Threshold = 0.5
    total = gunT.shape[0]+budoT.shape[0]
    proba = np.zeros((total,1))
    proba = np.array(proba)
    pred = np.zeros((total,1))
    pred = np.array(pred)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    return proba

def get_geoacc(estm_weights,gunT,budoT):
    Threshold = Find_Threshold(estm_weights,gunT,budoT)
    #Threshold = 0.5
    total = gunT.shape[0]+budoT.shape[0]
    proba = np.zeros((total,1))
    proba = np.array(proba)
    pred = np.zeros((total,1))
    pred = np.array(pred)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    gun_miss_sum=0
    budo_miss_sum=0
    for i in range(budoT.shape[0]):
        if proba[i+gunT.shape[0]] <= Threshold:
            budo_miss_sum= budo_miss_sum +1
            pred[i+gunT.shape[0]] = 0
        else:
            pred[i+gunT.shape[0]] = 1
    for i in range(gunT.shape[0]):
        if proba[i] >= Threshold:
            gun_miss_sum = gun_miss_sum+1
            pred[i] = 1
    err = gun_miss_sum +budo_miss_sum
    results = confusion_matrix(y_true=real,y_pred=pred)
    df = pd.DataFrame(results,columns=['건전 예측','부도 예측'],index=['실제 건전','실제 부도'])
    conma_temp = makeconma(conma,results)
    geo_acc = geoacc(conma)
    return geo_acc

def get_roc_auc_score(estm_weights,gunT,budoT):
    gunT = np.array(gunT)
    budoT = np.array(budoT)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    proba = np.zeros((gunT.shape[0]+budoT.shape[0],1))
    proba = np.array(proba)
    #print(gunT.shape)
    #exit(1)
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    return roc_auc_score(real,proba)

def get_gunT_budoT(dataset,estm_weights,n_estimators):
    budo_num = 0
    gun_num = 0
    for i in range(dataset[0].shape[0]):
        if dataset[0][i] == 0:
            gun_num = gun_num + 1
        else:
            budo_num = budo_num +1
    total = budo_num+gun_num

    gun = dataset.iloc[:gun_num,3]
    for i in range(n_estimators-1):
        gun = np.vstack([gun,dataset.iloc[:gun_num,3+2*(i+1)]])
    gun = gun.T
    budo = dataset.iloc[gun_num:,3]
    for i in range(n_estimators-1):
        budo = np.vstack([budo,dataset.iloc[gun_num:,3+2*(i+1)]])
    budo = budo.T
    #pred, real ,proba
    pred,real = predict(estm_weights, gun, budo)
    proba = predict_proba(estm_weights,gun,budo)
    return pred,real,proba

def AUROC_fit(n_estimators,dataset,max_epochs=100,learning_rate = 0.01,limit = 0.0005):
    optimal_estm_weights = np.zeros((n_estimators,1))
    estm_weights = np.zeros((n_estimators,1))

    for i in range(n_estimators):
        estm_weights[i] = 1/n_estimators
        optimal_estm_weights[i] = 1/n_estimators
        #dataset = pd.read_excel('AUROC2.xlsm')
        #print(dataset)
    budo_num = 0
    gun_num = 0
    for i in range(dataset[0].shape[0]):
        if dataset[0][i] == 0:
            gun_num = gun_num + 1
        else:
            budo_num = budo_num +1
    total = budo_num+gun_num

    gun = dataset.iloc[:gun_num,3] #

    for i in range(n_estimators-1):
        gun = np.vstack([gun,dataset.iloc[:gun_num,3+2*(i+1)]]) #
    #print(gun)
    #print(gun.shape)
    #exit(1)
    #gun = gun.T
    budo = dataset.iloc[gun_num:,3] #
    for i in range(n_estimators-1):
        budo = np.vstack([budo,dataset.iloc[gun_num:,3+2*(i+1)]]) #
    #budo = budo.T
    #budo = np.array(budo.T)
    #gun = np.array(gun.T)
    gun = np.array(gun)
    budo = np.array(budo)
    budo_average=np.zeros((n_estimators,1))
    gun_average=np.zeros((n_estimators,1))
    budo_cov = np.cov(budo)
    gun_cov = np.cov(gun)
    for i in range(n_estimators):
        gun_average[i]=gmean(gun[i,:])

    for i in range(n_estimators):
        budo_average[i]=gmean(budo[i,:])
    budo_gun_average = budo_average - gun_average
    #print(gun.T.shape)
    #exit(0)
    max_auroc = get_roc_auc_score(estm_weights,gun.T,budo.T)
    budo_gun_cov = ((gun_num-1)*gun_cov + (budo_num-1)*budo_cov)/ (budo_num+gun_num-2)
    epo = 0
    while True:
        WTU = np.dot(estm_weights.T, budo_gun_average)

        WTSW = np.dot(np.dot(estm_weights.T,budo_gun_cov),estm_weights)
        SQRT_WTSW = WTSW**(1/2)

        SW = np.dot(budo_gun_cov,estm_weights)

        expression_1 = (-1/((3.14*2)**(1/2)))*(np.exp(-0.5*((WTU/SQRT_WTSW)**2) ))

        expression_2 = np.dot(SQRT_WTSW,budo_gun_average.T)/WTSW

        expression_3 = WTU/SQRT_WTSW*SW/WTSW

        expression_4 = expression_1*(expression_2.T - expression_3)

        estm_weights = estm_weights-(learning_rate*expression_4)

        estm_weights = normalize_weight(estm_weights)



        if max_auroc < get_roc_auc_score(estm_weights,gun.T,budo.T):
            max_auroc = get_roc_auc_score(estm_weights,gun.T,budo.T)
            optimal_estm_weights = estm_weights
            #print("best auroc : ",max_auroc)
        epo +=1
        #print(epo)
        if max_epochs <= epo or SQRT_WTSW < limit :
            print("best epochs : ", epo)
            print("best auroc : ",max_auroc)
            print("optimal_weights",optimal_estm_weights)
            print("geoacc : ",get_geoacc(optimal_estm_weights,gun.T,budo.T))
            break
    return optimal_estm_weights

def ACCfit(dataset, max_epochs=10000, learning_rate=0.005, limit=0.0005, n_estimators=20):
    #print(dataset)
    #gun 몇개인지 budo 몇개인지 구하기
    budo_num = 0
    gun_num = 0
    for i in range(dataset[0].shape[0]):
        if dataset[0][i] == 0:
            gun_num = gun_num + 1
        else:
            budo_num = budo_num +1
    total = budo_num+gun_num

    gun = dataset.iloc[:gun_num,3] #
    for i in range(n_estimators-1):
        gun = np.vstack([gun,dataset.iloc[:gun_num,3+2*(i+1)]]) #
    gun = gun.T
    budo = dataset.iloc[gun_num:,3] #
    for i in range(n_estimators-1):
        budo = np.vstack([budo,dataset.iloc[gun_num:,3+2*(i+1)]]) #
    budo = budo.T
    budo = np.array(budo)
    gun = np.array(gun)

    #-----------------------------------------
    max_acc = 0
    optimal_estm_weights = []
    estm_weights = []
    for i in range(n_estimators):
        estm_weights.append(1/n_estimators)
        optimal_estm_weights.append(1/n_estimators)

    optimal_estm_weights = np.array(optimal_estm_weights)
    estm_weights = np.array(estm_weights)
    gun = np.array(gun)
    budo = np.array(budo)

    #for i in range(gun.shape[0]):
    #    for j in range(gun.shape[1]):
    #        gun[i][j] = gun[i][j]
    #for i in range(budo.shape[0]):
    #    for j in range(budo.shape[1]):
    #        budo[i][j] = budo[i][j]

    budo = budo.T
    budo_cov = np.cov(budo)
    budo_cov = np.round(budo_cov,10)
    gun=gun.T
    gun_cov = np.cov(gun)
    gun_cov = np.round(gun_cov,10)

    #for i in range(gun_cov.shape[0]):
    #    for j in range(gun_cov.shape[1]):
    #        gun_cov[i][j] = gun_cov[i][j]
    #for i in range(budo_cov.shape[0]):
    #    for j in range(budo_cov.shape[1]):
    #        budo_cov[i][j] = budo_cov[i][j]

    budo_average=np.zeros((n_estimators,1))
    gun_average=np.zeros((n_estimators,1))

    for i in range(n_estimators):
        gun_average[i]=gmean(gun[i,:])
    for i in range(n_estimators):
        budo_average[i]=gmean(budo[i,:])

    budoT=budo.T
    gunT =gun.T
    epochs = 0
    #get_proba(estm_weights,gunT,budoT)
    Threshold = Find_Threshold(estm_weights,gunT,budoT)

    #-----------------훈련 시작
    while True:
        budoT=budo.T
        gunT =gun.T
        epochs = epochs + 1

        gun_WTU = np.dot(estm_weights,gun_average)
        gun_WTSW = np.dot( np.dot(estm_weights,gun_cov), estm_weights)
        gun_SW = np.dot(gun_cov,estm_weights)
        gun_SQRT_WTSW = gun_WTSW**(1/2)
        gun_WSUM=np.dot(estm_weights,gun)

        gun_miss_sum=0
        budo_miss_sum=0

        for i in range(gun_num):
            if gun_WSUM[i] >= Threshold:
                gun_miss_sum = gun_miss_sum + 1

        gun_1 = np.exp(-0.5*((gun_WTU-Threshold)/gun_SQRT_WTSW)**2) #수정완료

        gun_2 = gun_SQRT_WTSW*(gun_average-Threshold)/gun_WTSW #수정완료
        gun_2 = gun_2.T[0]

        gun_3 = (((gun_WTU-Threshold)/gun_SQRT_WTSW)*gun_SW)/gun_WTSW #수정완료

        gun_4 = gun_1*(gun_2-gun_3)

        gun_5 = gun_4*(gun_miss_sum/gun_num)

        budo_WTU = np.dot(estm_weights,budo_average)
        budo_WTSW = np.dot( np.dot(estm_weights,budo_cov), estm_weights)

        budo_SW = np.dot(budo_cov,estm_weights)

        budo_SQRT_WTSW = budo_WTSW**(1/2)
        budo_WSUM=np.dot(estm_weights,budo)

        for i in range(budo_num):
            if budo_WSUM[i] <= Threshold:
                budo_miss_sum= budo_miss_sum +1

        budo_1 = np.exp(-0.5*((budo_WTU-Threshold)/budo_SQRT_WTSW)**2) #수정완료

        budo_2 = budo_SQRT_WTSW*(budo_average-Threshold)/budo_WTSW #수정완료
        budo_2 = budo_2.T[0]

        budo_3 = (((budo_WTU-Threshold)/budo_SQRT_WTSW)*budo_SW)/budo_WTSW #수정완료

        budo_4 = budo_1*(budo_2-budo_3)

        budo_5 = budo_4*(budo_miss_sum/budo_num)

        estm_weights = estm_weights - learning_rate*(gun_5 - budo_5)

        for i in range(estm_weights.shape[0]):
            estm_weights[i] = np.round(estm_weights[i],8)

        weight_total = 0
        for i in range(estm_weights.shape[0]):
            weight_total += estm_weights[i]

        for i in range(estm_weights.shape[0]):
            estm_weights[i] = estm_weights[i]/weight_total



        if(max_acc < get_proba(estm_weights,gunT,budoT)):
            optimal_estm_weights = estm_weights
            max_acc = get_proba(estm_weights,gunT,budoT)
            tmp = epochs

        if gun_SQRT_WTSW <= limit or budo_SQRT_WTSW <= limit or epochs >max_epochs :
            print("best epochs",tmp)
            break
        #---------------훈련 끝

    #gun_WSUM=np.dot(estm_weights,gun)
    #budo_WSUM=np.dot(estm_weights,budo)
    #gun_miss_sum=0
    #budo_miss_sum=0
    #for i in range(budo_num):
    #    if budo_WSUM[i] <= Threshold:
    #        budo_miss_sum= budo_miss_sum +1
    #for i in range(gun_num):
    #    if gun_WSUM[i] >= Threshold:
    #        gun_miss_sum = gun_miss_sum+1
    #print(optimal_estm_weights)
    return optimal_estm_weights

# 전역변수 데이터프레임 설정
global conmaA
conmaA=pd.DataFrame()
global conmaB
conmaB=pd.DataFrame()
global conmaC
conmaC=pd.DataFrame()

# confusion matrix 담을 것 + excel 출력 대상임.
conma=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])




def makecolname(n_fold,n_for):
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( 30 )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )
    return colname_first



def ToExcel(cv_acc,cv_auc,cv_geoacc,mat,colname_first,colname_second):
    accauc = np.array(cv_acc + cv_auc)
    geoaccs = np.array(cv_geoacc + cv_geoacc)
    mat = np.vstack((np.array(mat), accauc, geoaccs))
    mat = pd.DataFrame(np.array(mat),
                index = ['실제0','실제1','accauc','geoacc'],columns = arr)
    return mat



# AUC_fit 하는 것
def oncefit(xlsx_num, n_fold,epochs=1000):
    print("================================================ Fit 데이터크기",xlsx_num,"================================================")
    n_iter=0
    n_for = 3
    n_estimators=20

    #AUROC_fit(n_estimators,dataset,max_epochs=100,learning_rate = 0.01,limit = 0.0005):
    # 각 훈련에서 acc auc geoacc 담을 곳
    global conmaA
    conmaA=pd.DataFrame()
    global conmaB
    conmaB=pd.DataFrame()
    global conmaC
    conmaC=pd.DataFrame()
    global conmaD
    conmaD=pd.DataFrame()
    cv_acc_def=[]
    cv_auc_def=[]
    cv_acc_auc=[]

    cv_acc_defp=[] # default에다가 point만 추가했다는 뜻
    cv_auc_defp=[]
    cv_acc_aucp=[]

    cv_auc_auc=[]
    cv_acc_acc=[]
    cv_auc_acc=[]


    cv_geoacc_def = []
    cv_geoacc_auc = []
    cv_geoacc_acc = []

    cv_geoacc_defp = []
 #xlsx500 = pd.read_csv('resample_cc_500.csv')
#500
    # 데이터 프레임 컬럼만드는 것
    colname_first = makecolname(n_fold,n_for)
    f = open(str(xlsx_num)+".txt",'w')
    f.close()
    for i in range(n_for):
        print("================================================",i,"번째================================================")
        for j in range(10):
            # 이 지점에서 Adaboost 훈련이 끝
            # 이 지점에서 ACC Fit 훈련이 시작
            print("maximizeauroc fit start-----------",(i)*10+j+1)

            train_dataset = pd.read_excel('data/estimators'+str(int((xlsx_num+500)*9/10))+''+str((j+1)*(i+1))+'train.xlsx')
            test_dataset = pd.read_excel('data/estimators'+str(int((xlsx_num+500)/10))+''+str((j+1)*(i+1))+'test.xlsx')

            accweight = AUROC_fit(20,train_dataset,max_epochs=epochs,learning_rate=0.01,limit=0.0005)  #return  ACC_Weight_Vector
            #print(accweight.shape)
            pred,y_train,proba = get_gunT_budoT(train_dataset,accweight,n_estimators)
            mat=confusion_matrix(y_train,pred)
            conma_temp = makeconma(conma,mat)
            print(conma_temp)
            geoaccvalue = geoacc(conma_temp)
            print("geoacc = ",geoaccvalue)
            print("maximize auroc weight = ", accweight)
            pred,y_test,proba = get_gunT_budoT(test_dataset,accweight,n_estimators)

            f = open(str(xlsx_num)+".txt",'a')
            f.write( str(i*10+j+1) + "\n")
            dat = conma_temp
            f.write(str(dat))
            f.write('\n')
            dat = "geoacc = "+str(geoaccvalue)+'\n'
            f.write(dat)
            dat = "maximize_auroc_weight = "+str(accweight) +'\n'
            f.write(dat)
            f.write("\n\n\n")
            f.close()



            mat=confusion_matrix(y_test,pred)
            conma_temp = makeconma(conma,mat)

            # 이렇게하면 pred 까지 뽑아 낸 것임

            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc = roc_auc_score(y_test, proba)
            geoaccvalue = geoacc(conma_temp)

            cv_acc_acc.append(accuracy)
            cv_auc_acc.append(auc)
            cv_geoacc_acc.append(geoaccvalue)

            conmaC=pd.concat([conmaC,conma],axis=1)


    colname_second = conmaA.columns

    # ACC fit 부분 # 이거는 경민이꺼 받아서 써야함

    resultdef = ToExcel(cv_acc_acc,cv_auc_acc,cv_geoacc_acc,conmaC,colname_first,colname_second)
    resultdef.to_excel('Result/maxmize_AUROC_fit'+str(xlsx_num) + '.xlsx')

    print("================================================종료================================================")
