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


def predict(estm_weights, gunT, budoT):
    #print("gun: ",gunT.shape)
    #print("budo: ",budoT.shape)
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
    #print(Threshold)
    #print(proba)
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
    #print("target",target)
    #print("pred",predicted)
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index = i), 'threshold' : pd.Series(threshold, index = i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def Find_Threshold(estm_weights,gunT,budoT):
    #print("gun:",gunT)
    #print("budo:",budoT)
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
    #print(gunT.shape)
    #print("estm",estm_weights)
    #print("proba",proba)
    #print("gunT",gunT)
    #print("budoT",budoT)
    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    #print("proba:",proba)
    return Find_Optimal_Cutoff(real,proba)

def get_proba(estm_weights,gunT,budoT):
    #print("gun: ",gunT.shape)
    #print("budo: ",budoT.shape)
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
    #print("real:",real.shape)
    #print("pred:",pred.shape)

    #print("건전 못맞춤 : ",gun_miss_sum)
    #print("부도 못맞춤 : ",budo_miss_sum)
    results = confusion_matrix(y_true=real,y_pred=pred)
    #print("실제/예측")
    df = pd.DataFrame(results,columns=['건전 예측','부도 예측'],index=['실제 건전','실제 부도'])
    #print(df)
    conma_temp = makeconma(conma,results)
    geo_acc = geoacc(conma)

    return geo_acc
    #return 1-err/total 단순 산술 평균

def get_roc_auc_score(estm_weights,gunT,budoT):
    gunT = np.array(gunT)
    budoT = np.array(budoT)
    real_gun = np.zeros((gunT.shape[0],1))
    real_budo = np.zeros((budoT.shape[0],1))
    real_budo = real_budo+1
    real = np.vstack([real_gun,real_budo])
    proba = np.zeros((gunT.shape[0]+budoT.shape[0],1))
    proba = np.array(proba)

    for i in range(gunT.shape[0]):
        for j in range(gunT.shape[1]):
            proba[i] = proba[i]+ gunT[i][j]*estm_weights[j]

    for i in range(budoT.shape[0]):
        for j in range(budoT.shape[1]):
            proba[gunT.shape[0]+i] = proba[gunT.shape[0]+i]+ budoT[i][j]* estm_weights[j]
    return roc_auc_score(real,proba)

#max_epochs = 10000, learning_rate = 0.001
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
    #gun = dataset.iloc[:gun_num,1:n_estimators+1]
    #budo = dataset.iloc[gun_num:total,1:n_estimators+1]
    #gun = np.array(gun)
    #budo = np.array(budo)
    #for i in range(gun.shape[0]):
    #    for j in range(gun.shape[1]):
    #        gun[i][j]=gun[i][j]
    #for i in range(budo.shape[0]):
    #    for j in range(budo.shape[1]):
    #        gun[i][j]=[i][j]
    #-----------------------------------------
    max_acc = 0
    optimal_estm_weights = []
    estm_weights = []
    for i in range(n_estimators):
        estm_weights.append(1/n_estimators)
        optimal_estm_weights.append(1/n_estimators)

    optimal_estm_weights = np.array(optimal_estm_weights)
    estm_weights = np.array(estm_weights)
    #gun = dataset.iloc[:450,0:n_estimators]
    #budo = dataset.iloc[450:901,0:n_estimators]
    gun = np.array(gun)
    budo = np.array(budo)
    #learning_rate = 0.001

    for i in range(gun.shape[0]):
        for j in range(gun.shape[1]):
            gun[i][j] = gun[i][j]
    for i in range(budo.shape[0]):
        for j in range(budo.shape[1]):
            budo[i][j] = budo[i][j]
    budo = budo.T
    budo_cov = np.cov(budo)
    budo_cov = np.round(budo_cov,10)
    gun=gun.T
    gun_cov = np.cov(gun)
    gun_cov = np.round(gun_cov,10)
    #budo_cov = np.round(budo_cov,7)
    #gun_cov = np.round(gun_cov,7)
    #print("cov",gun_cov)
    #print("cov",budo_cov)

    for i in range(gun_cov.shape[0]):
        for j in range(gun_cov.shape[1]):
            gun_cov[i][j] = gun_cov[i][j]
    for i in range(budo_cov.shape[0]):
        for j in range(budo_cov.shape[1]):
            budo_cov[i][j] = budo_cov[i][j]
    budo_average=np.zeros((n_estimators,1))
    gun_average=np.zeros((n_estimators,1))

    for i in range(n_estimators):
        gun_average[i]=gmean(gun[i,:])
    for i in range(n_estimators):
        budo_average[i]=gmean(budo[i,:])


    budoT=budo.T
    gunT =gun.T
    epochs = 0
    #print("훈련 전")
    get_proba(estm_weights,gunT,budoT)
    Threshold = Find_Threshold(estm_weights,gunT,budoT)

    #print("th",Threshold)
    #print("훈련 후")
    #-----------------훈련 시작
    while True:
        #print(111111111111111)
        #print("gun",gun)
        budoT=budo.T
        gunT =gun.T
        #print("Threshold",Threshold)
        epochs = epochs + 1

        gun_WTU = np.dot(estm_weights,gun_average)
        gun_WTSW = np.dot( np.dot(estm_weights,gun_cov), estm_weights)
        gun_SW = np.dot(gun_cov,estm_weights)
        gun_SQRT_WTSW = gun_WTSW**(1/2)
        gun_WSUM=np.dot(estm_weights,gun)

        #gun_WTU = np.round(gun_WTU,5)
        #gun_WTSW = np.round(gun_WTSW,5)
        #gun_SW = np.round(gun_SW,5)
        #gun_SQRT_WTSW = np.round(gun_SQRT_WTSW,5)
        #gun_WSUM = np.round(gun_WSUM,5)

        gun_miss_sum=0
        budo_miss_sum=0
        gun_pred=[]
        gun_mis=[]
        #print(gun_WSUM)
        for i in range(gun_num):
            if gun_WSUM[i] >= Threshold:
                gun_miss_sum = gun_miss_sum + 1


        gun_1 = np.exp(-0.5*((gun_WTU-Threshold)/gun_SQRT_WTSW)**2) #수정완료
        #gun_1 = np.round(gun_1,6)
        #print(gun_1)
        gun_2 = gun_SQRT_WTSW*(gun_average-Threshold)/gun_WTSW #수정완료
        #gun_2 = np.round(gun_2,6)
        gun_2 = gun_2.T[0]
        #print(gun_2)
        gun_3 = (((gun_WTU-Threshold)/gun_SQRT_WTSW)*gun_SW)/gun_WTSW #수정완료
        #gun_3 = np.round(gun_3,6)
        #print(gun_3)
        gun_4 = gun_1*(gun_2-gun_3)
        #gun_4 = np.round(gun_4,6)
        #print(gun_4)
        gun_5 = gun_4*(gun_miss_sum/gun_num)
        #gun_5 = np.round(gun_5,6)

        budo_WTU = np.dot(estm_weights,budo_average)
        budo_WTSW = np.dot( np.dot(estm_weights,budo_cov), estm_weights)
        #print("budo_cov",budo_cov)
        #print("estm_weights",estm_weights)
        #print("bdWTSW",budo_WTSW)
        budo_SW = np.dot(budo_cov,estm_weights)

        budo_SQRT_WTSW = budo_WTSW**(1/2)
        budo_WSUM=np.dot(estm_weights,budo)

        #budo_WTU = np.round(budo_WTU,5)
        #budo_WTSW = np.round(budo_WTSW,5)
        #budo_SW = np.round(budo_SW,5)
        #budo_SQRT_WTSW = np.round(budo_SQRT_WTSW,5)
        #budo_WSUM = np.round(budo_WSUM,5)

        #print("gunT",gunT)
        #print("budoT",budoT)
        #print("budoWtu",budo_WTU)
        #print("budoWtsw",budo_WTSW)
        #print("estm",estm_weights)
        #print("budosw",budo_SW)
        #print("budoWsum",budo_WSUM)
        for i in range(budo_num):
            if budo_WSUM[i] <= Threshold:
                budo_miss_sum= budo_miss_sum +1
        #print("estm_weights",estm_weights)
        #print("gun_sum",gun_WSUM)
        #print("budo_sum",budo_WSUM)

        budo_1 = np.exp(-0.5*((budo_WTU-Threshold)/budo_SQRT_WTSW)**2) #수정완료
        #budo_1 = np.round(budo_1,6)
        #print(budo_1)
        budo_2 = budo_SQRT_WTSW*(budo_average-Threshold)/budo_WTSW #수정완료
        #budo_2 = np.round(budo_2,6)
        budo_2 = budo_2.T[0]
        #print(budo_2)
        budo_3 = (((budo_WTU-Threshold)/budo_SQRT_WTSW)*budo_SW)/budo_WTSW #수정완료
        #budo_3 = np.round(budo_3,6)
        #print(budo_3)
        budo_4 = budo_1*(budo_2-budo_3)
        #budo_4 = np.round(budo_4,6)
        budo_5 = budo_4*(budo_miss_sum/budo_num)
        #budo_5 = np.round(budo_5,6)
        #print(gun_miss_sum)
        #print("before",estm_weights)
        #print(budo_5)
        #print(gun_5)
        #print("budo3",budo_3)
        #print("budo4",budo_4)
        #print("budo miss sum",budo_miss_sum)
        #print("gun miss sum",gun_miss_sum)
        estm_weights = estm_weights - learning_rate*(gun_5 - budo_5)
        for i in range(estm_weights.shape[0]):
            estm_weights[i] = np.round(estm_weights[i],8)
        s=0
        for i in estm_weights:
            s = s+i
        short = 1-s
        tip = short/n_estimators
        for i in range(estm_weights.shape[0]):
            estm_weights[i] = estm_weights[i]+tip
        sum =0
        for i in estm_weights:
            sum = sum + i
        #print("sum : ",sum)
        #print("aft",estm_weights)
        #print(gun_miss_sum)
        #print(budo_miss_sum)
        if(max_acc < get_proba(estm_weights,gunT,budoT)):
            optimal_estm_weights = estm_weights
            max_acc = get_proba(estm_weights,gunT,budoT)
            tmp = epochs
        #print(budo_SQRT_WTSW)
        #print(gun_SQRT_WTSW)

        if gun_SQRT_WTSW <= limit or budo_SQRT_WTSW <= limit or epochs >max_epochs :
            #print(budo_SQRT_WTSW)
            #print(gun_SQRT_WTSW)
            print("best epochs",tmp)
            break
        #print(epochs)
        #---------------훈련 끝

    gun_WSUM=np.dot(estm_weights,gun)
    budo_WSUM=np.dot(estm_weights,budo)
    gun_miss_sum=0
    budo_miss_sum=0
    for i in range(budo_num):
        if budo_WSUM[i] <= Threshold:
            budo_miss_sum= budo_miss_sum +1
    for i in range(gun_num):
        if gun_WSUM[i] >= Threshold:
            gun_miss_sum = gun_miss_sum+1

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

# roc 곡선 그리기 위한 함수
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림.
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림.
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

#AUC + confusion mat
# 교수님 주신 논문에서 만들었으나 아직 사용은 X 나중에 필요시 사용


def new_auc(y_test,pred):
    conma=confusion_matrix(y_test , pred) # y_test부분에 실제값, 2번째 파라미터로 인자값
    TPR = conma[1,1]/(conma[1,0]+conma[1,1])
    FPR = conma[0,1]/(conma[0,0]+conma[0,1])
    return (1+TPR-FPR)/2

def Maketable(clf,X,y): # X = X_train, y = y_train
    table = pd.DataFrame(np.array(y)) # 그냥 y하면 데이터프레임이라서 기업 인덱스가 존재함
    for i, estimator in enumerate(clf.estimators_):
        ap = pd.DataFrame(estimator.predict(X), columns = ['0_'+str(i+1),'1_'+str(i+1)] ) # 이게 아마 넘파이 일거고
        table = pd.concat([table,ap],axis=1)
    return table

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def makeexcel(clf,X_train,X_test,y_train,y_test):
    for i in range(len(clf.estimators_)):
        X = pd.DataFrame()
        y = pd.DataFrame()
        proba1 = clf.estimators_[i].predict(X_train)
        proba2 = clf.estimators_[i].predict(X_test)
        proba = pd.DataFrame(np.vstack((proba1,proba2)))
        X = pd.concat([X_train,X_test],ignore_index = True)
        y = pd.concat([y_train,y_test],ignore_index = True)
        indexname = ['train']*int((len(X)*(9/10)))+['test']*int(len(X)/10)
        result = pd.concat([X,y,proba],axis=1,ignore_index = True)
        result.index = indexname
        result.to_excel('estimators/first_fold_estimators'+str(i+1)+'.xlsx')




def acccondition(clf):
    clf.accfit = True
    clf.newfit = False
    clf.weight = False


def makecolname(n_fold,n_for):
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( 30 )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )
    return colname_first



def ToExcel(cv_acc,cv_auc,cv_geoacc,mat,colname_first,colname_second):
    accauc = np.array(cv_acc + cv_auc)
    geoaccs = np.array(cv_geoacc + cv_geoacc)
    #print(mat)
    #print(accauc)
    #print(geoaccs)
    mat = np.vstack((np.array(mat), accauc, geoaccs))
    #mat = pd.DataFrame(np.array(mat),
    #            index = ['실제0','실제1','accauc','geoacc'],
    #            columns = [colname_first, colname_second])


    mat = pd.DataFrame(np.array(mat),
                index = ['실제0','실제1','accauc','geoacc'],columns = arr)
    return mat



# AUC_fit 하는 것
def oncefit(xlsx_num, n_fold,epochs=1000):
    print("================================================ Fit 데이터크기",xlsx_num,"================================================")
    #skfold = StratifiedKFold(n_splits=n_fold)
    n_iter=0
    n_for = 3
    n_estimators=20


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
            print("ACC fit start-----------",(i)*10+j+1)
            #clf.table = Maketable(clf,X_train, y_train)
            #clf.table.to_excel(excel_writer = '1.xlsx')
            #acccondition(clf)
            #train -> xlsx_num+500 *0.9
            #test -> (xlsx_num+500)*0.1
            train_dataset = pd.read_excel('data/estimators'+str(int((xlsx_num+500)*9/10))+''+str((j+1)*(i+1))+'train.xlsx')
            test_dataset = pd.read_excel('data/estimators'+str(int((xlsx_num+500)/10))+''+str((j+1)*(i+1))+'test.xlsx')

            accweight = ACCfit(train_dataset,max_epochs=epochs,learning_rate=0.005,limit=0.0005,n_estimators=20)  #return  ACC_Weight_Vector
            pred,y_train,proba = get_gunT_budoT(train_dataset,accweight,n_estimators)
            mat=confusion_matrix(y_train,pred)
            conma_temp = makeconma(conma,mat)
            print(conma_temp)
            geoaccvalue = geoacc(conma_temp)
            print("geoacc = ",geoaccvalue)
            print("accweight = ", accweight)
            pred,y_test,proba = get_gunT_budoT(test_dataset,accweight,n_estimators)

            f = open(str(xlsx_num)+".txt",'a')
            f.write( str(i*10+j+1) + "\n")
            dat = conma_temp
            f.write(str(dat))
            f.write('\n')
            dat = "geoacc = "+str(geoaccvalue)+'\n'
            f.write(dat)
            dat = "accweight = "+str(accweight) +'\n'
            f.write(dat)
            f.write("\n\n\n")
            f.close()


            #pred = clf.predict(X_test,Threshold)

            mat=confusion_matrix(y_test,pred)
            conma_temp = makeconma(conma,mat)

            # 이렇게하면 pred 까지 뽑아 낸 것임
            #print("ytest ",y_test)
            #print("proba",proba)
            #print("pred", pred)

            accuracy = np.round(accuracy_score(y_test,pred), 4)
            #print("accuracy",accuracy)
            auc = roc_auc_score(y_test, proba)
            geoaccvalue = geoacc(conma_temp)

            cv_acc_acc.append(accuracy)
            cv_auc_acc.append(auc)
            cv_geoacc_acc.append(geoaccvalue)
            #print(accuracy)

            #print("\n")
            #print(n_iter,conma_temp,"\n","acc:",accuracy,"auc:",auc)
            conmaC=pd.concat([conmaC,conma],axis=1)
            # 이 지점에서 ACC pred이 끝
            #exit(0)

    colname_second = conmaA.columns

    # ACC fit 부분 # 이거는 경민이꺼 받아서 써야함

    resultdef = ToExcel(cv_acc_acc,cv_auc_acc,cv_geoacc_acc,conmaC,colname_first,colname_second)
    resultdef.to_excel('Result/ACCfit'+str(xlsx_num) + '.xlsx')

    print("================================================종료================================================")

def readexcel(roadname, datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_excel(roadname+str(c)+'.xlsx'))
    return Datalist

def readcsv(roadname ,datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_csv(roadname+str(c)+'.csv'))
    return Datalist

def valaccauc(datas):
    for data in datas :
        reslen = int((data.shape[1]) / 2)
        accs = np.array(data.iloc[4,1:reslen+1])
        aucs = np.array(data.iloc[4,reslen+1:data.shape[1]])
        geoaccs = np.array(data.iloc[5,1:reslen+1])

        print( '-' * 30 )
        print(' acc_mean : %.2f \n acc_std : %.2f \n auc_mean : %.2f \n auc_std : %.2f \n geoacc_mean : %.2f \n geoacc_std : %.2f '
              % (np.mean(accs),np.std(accs), np.mean(aucs), np.std(aucs), np.mean(geoaccs), np.std(geoaccs)) )
        print( '-' * 30 )
