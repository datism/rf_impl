from array import array
import math
import time
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)

class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs, idxs, depth = 10, min_leaf = 5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n = len(idxs)
        self.val = np.mean(y[idxs])
        self.score = float('inf')

        self.find_varsplit()
        
    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)

        if self.is_leaf: 
            return

        #lấy dữ liệu của cột được chọn để tach
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]

        #random index cho thuộc tinh
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]

        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1, min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1, min_leaf=self.min_leaf)

    def find_better_split(self, var_idx):
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]

        #phần bên phải
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        #phần bên trái
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0, self.n - self.min_leaf - 1):
            #lấy phần tử bên phải chuyển sang bên trái
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2

            if i < self.min_leaf or xi == sort_x[i+1]:
                continue
            
            #tính độ lệch chuẩn của 2 phần
            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt

            if curr_score<self.score: 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi
    
    @property
    def split_col(self): return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf') or self.depth <= 0 
    
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)


class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_sz, depth = 10, min_leaf = 5):
        np.random.seed(12)

        #số thuộc tính được truyền cho mỗi cây con
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
    
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf  = x, y, sample_sz, depth, min_leaf
        
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        #random index cho dữ liệu train
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        #random index cho thuộc tính
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]

        return DecisionTree(self.x[idxs], self.y[idxs], self.n_features, f_idxs,
                    idxs=np.array(range(self.sample_sz)), depth = self.depth, min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis = 0)



# đọc dữ liệu train      
with open('data/diabetes_train.pkl', 'rb') as f:
    diabetes_train = pickle.load(f)
print("Số chiều input: ", diabetes_train['data'].shape)
print("Số chiều target y tương ứng: ", diabetes_train['target'].shape)
print()

# print("2 mẫu dữ liệu đầu tiên:")
# print("input: ", diabetes_train['data'][:2])
# print("target: ", diabetes_train['target'][:2])
# print()
trainTime = [0, 0]
predTime = [0, 0]
rmse = [0, 0]

# Train
start = time.time()
diabetesTree = RandomForest(diabetes_train['data'], diabetes_train['target'], 100, 'sqrt', diabetes_train['data'].shape[0])
trainTime[0] = time.time() - start

start = time.time()
diabetes1Tree = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5)
diabetes1Tree.fit(diabetes_train['data'], diabetes_train['target'])
trainTime[1] = time.time() - start

# đọc dữ liệu test
# dữ liệu test có cấu trúc giống dữ liệu huấn luyện nhưng số lượng mẫu chỉ là 42
with open('data/diabetes_test.pkl', 'rb') as f:
    diabetes_test = pickle.load(f)

# Thực hiện phán đoán cho dữ liệu mới
start = time.time()
diabetes_y_pred = diabetesTree.predict(diabetes_test['data'])
predTime[0] = time.time() - start

start = time.time()
diabetes_y_pred1 = diabetes1Tree.predict(diabetes_test['data'])
predTime[1] = time.time() - start

# In ra 5 phán đoán đầu tiên
print("-----5 phán đoán đầu tiên-----")
df = pd.DataFrame(data=np.array([diabetes_test['target'], diabetes_y_pred,
                            abs(diabetes_test['target'] - diabetes_y_pred)]).T,
             columns=["y thực tế", "y dự đoán", "Lệch"])

print(df.head(5))
print()

# Tính RMSE
rmse[0] = math.sqrt(mean_squared_error(diabetes_test['target'], diabetes_y_pred))
rmse[1] = math.sqrt(mean_squared_error(diabetes_test['target'], diabetes_y_pred1))

# So sánh hai mô hình
print("-----So sánh hai mô hình-----")
cmp = pd.DataFrame({'thời gian train': trainTime, 'thời gian dự đoán': predTime, 'rmse': rmse},
                        index={'RF', 'RFR'})
print(cmp)
print()

# Biểu đồ phân phối đầu ra
plt.figure(figsize=(8, 8))

plt.subplot(311)
plt.title("Phân phối của đầu ra thực tế")
sns.histplot(diabetes_test['target'])

plt.subplot(312)
plt.title("Phân phối dự đoán đầu ra của mô hình RandomForest")
sns.histplot(diabetes_y_pred)

plt.subplot(313)
plt.title("Phân phối dự đoán đầu ra của mô hình RandomForestRegressor")
sns.histplot(diabetes_y_pred1)

plt.subplots_adjust(top=0.95, hspace=0.55)
plt.show()