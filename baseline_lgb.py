import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================
# 1. 读取数据
# ===============================
train = pd.read_csv('data/train_format1.csv')
test = pd.read_csv('data/test_format1.csv')

# 日志：只读必要列，避免内存炸
log = pd.read_csv(
    'data/user_log_format1.csv',
    usecols=['user_id', 'seller_id', 'action_type']
)

# 统一字段名
log = log.rename(columns={'seller_id': 'merchant_id'})

print('数据读取完成')
print('train:', train.shape)
print('test:', test.shape)
print('log:', log.shape)

# ===============================
# 2. 行为特征统计函数
# ===============================
def behavior_count(log_df, action_type, feature_name):
    tmp = log_df[log_df['action_type'] == action_type]
    return (
        tmp.groupby(['user_id', 'merchant_id'])
           .size()
           .reset_index(name=feature_name)
    )

# ===============================
# 3. 统计四类行为
# ===============================
click_cnt = behavior_count(log, 0, 'click_cnt')   # 点击
cart_cnt  = behavior_count(log, 1, 'cart_cnt')    # 加购
buy_cnt   = behavior_count(log, 2, 'buy_cnt')     # 购买
fav_cnt   = behavior_count(log, 3, 'fav_cnt')     # 收藏

print('行为特征统计完成')

# ===============================
# 4. 合并特征
# ===============================
for feat_df in [click_cnt, cart_cnt, buy_cnt, fav_cnt]:
    train = train.merge(feat_df, on=['user_id', 'merchant_id'], how='left')
    test = test.merge(feat_df, on=['user_id', 'merchant_id'], how='left')

# 缺失值补 0
for col in ['click_cnt', 'cart_cnt', 'buy_cnt', 'fav_cnt']:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

print('特征合并完成')

# ===============================
# 5. 构造训练数据
# ===============================
features = ['click_cnt', 'cart_cnt', 'buy_cnt', 'fav_cnt']

X = train[features]
y = train['label']
X_test = test[features]

# ===============================
# 6. 切分验证集
# ===============================
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7. 训练 LightGBM
# ===============================
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_tr, y_tr)

# ===============================
# 8. 验证集评估
# ===============================
val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print(f'Validation AUC: {auc:.4f}')

# ===============================
# 9. 测试集预测 & 提交
# ===============================
test['prob'] = model.predict_proba(X_test)[:, 1]

submission = test[['user_id', 'merchant_id', 'prob']]
submission.to_csv('submission.csv', index=False)

print('LightGBM submission.csv 已生成（4 行为特征版）')
