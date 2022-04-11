# 工作总结



## 需求： 找出异常商品（价格异常、销量异常）



## 实际情况：

1. 已知商品价格、销量、收藏数、评论数、店铺评分等指标，且部分收藏和评论数数值缺失。

2. 企业只提供了几个月份的总数据，并没有告知哪些数据是异常的，因此无法使用监督学习进行异常检测。**故侧重点在无监督学习的异常检测。**



## 算法和模型训练

我们选择的主要工具是pycaret，其内置无监异常检测，并且可以自动标记异常数据和生成模型。介绍如下：

PyCaret’s Anomaly Detection Module is an unsupervised machine learning module that is used for identifying rare items, events or observations which raise suspicions by differing significantly from the majority of the data. Typically, the anomalous items will translate to some kind of problem such as bank fraud, a structural defect, medical problems or errors . This module provide several pre-processing features that prepares the data for modeling through setup function. This module has over 12 ready-to-use algorithms and several plots to analyze the results of trained models

常见的无监督检测方法如下：

|           |                              Name |          Reference          |
| --------: | --------------------------------: | :-------------------------: |
|        ID |                                   |                             |
|      abod |      Angle-base Outlier Detection |    pyod.models.abod.ABOD    |
|   cluster |    Clustering-Based Local Outlier |   pyod.models.cblof.CBLOF   |
|       cof |  Connectivity-Based Local Outlier |     pyod.models.cof.COF     |
|   iforest |                  Isolation Forest | pyod.models.iforest.IForest |
| histogram | Histogram-based Outlier Detection |    pyod.models.hbos.HBOS    |
|       knn |      K-Nearest Neighbors Detector |     pyod.models.knn.KNN     |
|       lof |              Local Outlier Factor |     pyod.models.lof.LOF     |
|       svm |            One-class SVM detector |   pyod.models.ocsvm.OCSVM   |
|       pca |      Principal Component Analysis |     pyod.models.pca.PCA     |
|       mcd |    Minimum Covariance Determinant |     pyod.models.mcd.MCD     |
|       sod |        Subspace Outlier Detection |     pyod.models.sod.SOD     |
|       sos |      Stochastic Outlier Selection |     pyod.models.sos.SOS     |

针对已知数据我们选取了**Clustering-Based局部离群值检测、Connectivity-Based局部离群值、隔离森林检测、knn检测、lof离群值检测**。

部分代码如下：基本思路为从总数据集中导出一个样本，针对这个样本对数据进行训练并得到不同检测算法的模型，将模型人工再此修正后重新训练模型，并把模型用于总数据集得到最终结果。

```python
## 训练模型
## 载入的训练数据需要以 数字_trainData 的形式命名
def AnomalyDetection():
   for i in range(200):
        ##———————————————————导入数据——————————————————————##
        all_datasets = get_data(str(i)+'_trainData')

        data = all_datasets.sample(frac=0.97, random_state=810)
        data_unseen = all_datasets.drop(data.index)
        ##———————————————————处理数据——————————————————————##
        data.reset_index(drop=True, inplace=True)
        data_unseen.reset_index(drop=True, inplace=True)
        exp_ano101 = setup(all_datasets, normalize = True, 
                        ignore_features = ['商品ID','店铺ID','三级类目名','收藏数','评论数','收藏评论综合2','收藏评论综合','店铺评分'],
                        session_id = 279)
        ##———————————————————建立模型——————————————————————##
        iforest = create_model('iforest')   
        print(iforest)
        ##———————————————————开始预测——————————————————————##
        results = assign_model(iforest)
        results.head()
        data_predictions = predict_model(iforest, data = all_datasets)
        data_predictions.head()
        ##———————————————————保存模型——————————————————————##
        save_model(iforest,str(i)+'_Iforest')
        ##———————————————————输出处理结果到csv文件——————————————————————##
        data_predictions.to_csv(str(i)+'_Iforest'+".csv",index=False,sep=',')

## 载入模型并处理数据
## 载入的预测数据需要以 数字_realData 的形式命名
def LoadModel(i):
    all_datasets = get_data(str(i)+'_realData')
    save_model = load_model(str(i)+'_Iforest')
    new_predictions = predict_model(save_model, data = all_datasets)
    ##———————————————————输出处理结果到csv文件——————————————————————##
    data_predictions.to_csv(str(i)+'_Iforest_Predicted'+".csv",index=False,sep=',')
 
 
    
# 按照三级类目分成200个文件，按照0-199进行编号
def main():
    AnomalyDetection() # 训练模型
    for i in range(200):
        LoadModel(i)   # 载入模型并处理数据
```

得到的模型如下：

![image-20220320221028804](C:\Users\29185\AppData\Roaming\Typora\typora-user-images\image-20220320221028804.png)

检测出来的部分异常数据如下：

![image-20220320221111424](C:\Users\29185\AppData\Roaming\Typora\typora-user-images\image-20220320221111424.png)



## 算法和模型的进一步改进

  异常检测有三个特点：

1. 无监督：一般没有标签可用

2. 极端的数据不平衡：异常点往往远少于正常点

3. 复杂的模式

  在这种前提下，**人们一般会使用集成学习来解决这个问题，也就是训练一大堆异常检测模型**，**再去合并**。因为我们说到异常检测是无监督，人们一般会同时独立训练模型（像bagging），而非做依赖顺序模型（sequential），因为后者不好评估。所以绝大部分的异常检测集成都是这种并行式的（parallel）。

  人们会用简单的平均、加权平均或者求最大值来做。但这一类简单粗暴的合并方式有两个问题：

- **不存在选择的过程（no selection process），因此只能得到平庸的结果**
- **忽视了异常检测的局部性特征**

100个训练好的基础模型 ![[公式]](https://www.zhihu.com/equation?tex=D_1%2C+D_2%2C...%2CD_%7B100%7D) ，我有新的数据 ![[公式]](https://www.zhihu.com/equation?tex=X) 想用它们进行预测是否是异常，那我们可以对于 ![[公式]](https://www.zhihu.com/equation?tex=X_1) 用 ![[公式]](https://www.zhihu.com/equation?tex=D_1%2BD_3%2BD_4) 的综合结果，然后对 ![[公式]](https://www.zhihu.com/equation?tex=X_2) 用 ![[公式]](https://www.zhihu.com/equation?tex=D_1%2BD_5%2BD_8%2BD_%7B10%7D) 来预测。

很自然的想到一种在监督学习中会用的方法Dynamic Classifier Selection （动态分类器选择）。它的根本思想就是，每当我们获得一个新的![[公式]](https://www.zhihu.com/equation?tex=X)进行预测时，先评估一下哪些基学习器（base classifiers）在这个点附近的区域上表现良好（也就是在![[公式]](https://www.zhihu.com/equation?tex=X)的邻近区域上），那我们就更可以相信它会在![[公式]](https://www.zhihu.com/equation?tex=X)上表现良好。所以最简单的就是对于![[公式]](https://www.zhihu.com/equation?tex=X)，找到在它附近的训练数据上表现最好的模型 ![[公式]](https://www.zhihu.com/equation?tex=C_i+) ，然后输出 ![[公式]](https://www.zhihu.com/equation?tex=C_i%28X%29) 作为![[公式]](https://www.zhihu.com/equation?tex=X)的结果即可。这种思路其实也很适合异常检测问题，因为考虑“附近区域”就是一种考虑局部关系的方法。**所以如果可以把DCS移植到异常检测的语境下，那就同时解决了“选择”和“局部性”两个问题。**

  针对没有标签和模型选择的问题，我们提出下面三条措施：

1. **生成伪标签来进行评估**。我们使用所有基学习器的输出结果的{均值，最大值}作为伪标签。衡量模型在小范围的上的表现就是求在那个区域上 ![[公式]](https://www.zhihu.com/equation?tex=D_i) 的输出和伪标签之间的Pearson Correlation。
2. **定义![[公式]](https://www.zhihu.com/equation?tex=X)的附近区域（邻居)时采用了一种随机K近邻的方法来提高稳定性，降低维数灾难的影响**。
3. 在![[公式]](https://www.zhihu.com/equation?tex=X)的最终结果时，**选择多个表现优异的模型**，**进行二次合并而非依赖于单一的模型输出**，提高表现。

综上，最终的LSCP有四个部分：

1. **训练多个基础异常检测器**（Base Detector Generation）
2. **生成伪标签用于评估**（Pseudo Ground Truth）
3. **对于每个测试点生成局部空间**，也就是近邻（Local Region Definition）
4. **模型选择与合并**（Model Selection and Combination），即**对所有的基模型在我们找到的局部空间上用生产的伪标签进行评估**，和伪标签在局部空间上Pearson大的被选做最终输出模型。

在以上一系列过程中，为了提升效果，我们做了两种交叉选择：

1. 用均值（A）或者最大值（M）作为伪标签
2. 最终依赖于单个最优模型，还是二次合并选择 ![[公式]](https://www.zhihu.com/equation?tex=s) 个模型再合并一次

最终得到了4个LSCP模型，分别是*LSCP_A*，*LSCP_M*，*LSCP_AOM*，*LSCP_MOA*，前两个只选择1个模型，而后两个会有二次合并过程。



##  改进模型的实验结果

![img](https://pic4.zhimg.com/v2-01c4479d6e528ba06b8648b137bd9757_r.jpg)

![img](https://pic4.zhimg.com/v2-1691fe35f6ac8616654e697605cbdfdb_r.jpg)

我们对提出的4种LSCP方法与传统的7种合并方法在20个异常检测数据上进行对比（ROC和mAP）,我们发现其中的*LSCP_AOM*的效果非常稳定，在大部分情况下都有不俗的表现。

  我们使用了PyOD进行进一步检测，PyOD描述如下：

PyOD is a comprehensive and scalable **Python toolkit** for **detecting outlying objects** in multivariate data. This exciting yet challenging field is commonly referred as [Outlier Detection](https://en.wikipedia.org/wiki/Anomaly_detection) or [Anomaly Detection](https://en.wikipedia.org/wiki/Anomaly_detection).

PyOD includes more than 30 detection algorithms, from classical LOF (SIGMOD 2000) to the latest SUOD (MLSys 2021) and ECOD (TKDE 2022). Since 2017, PyOD has been successfully used in numerous academic researches and commercial products [[35\]](https://github.com/yzhao062/pyod#zhao2019lscp) [[36\]](https://github.com/yzhao062/pyod#zhao2021suod) with more than 5 million downloads. It is also well acknowledged by the machine learning community with various dedicated posts/tutorials, including [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/), [KDnuggets](https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html), [Towards Data Science](https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1), [Computer Vision News](https://rsipvision.com/ComputerVisionNews-2019March/18/), and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning#python-general-purpose).

  测试案例：

```python
# -*- coding: utf-8 -*-
"""Example of using LSCP for outlier detection
"""
# Author: Zain Nasrullah <zain.nasrullah.zn@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, y_train, X_test, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      contamination=contamination,
                      random_state=42)
    X_train, X_test = standardizer(X_train, X_test)

    # train lscp
    clf_name = 'LSCP'
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35)]
    clf = LSCP(detector_list, random_state=42)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
```





## 展望

1.处理确实数据，使用更多的指标预测

2.多模型结合处理

3.得到正确数据进一步训练模型