项目名称：天池电商用户复购预测（Baseline 实现）

一、项目说明
本项目基于天池官方提供的用户行为数据，构建用户—商家层面的复购预测模型。
任务目标为判断给定的 (user_id, merchant_id) 是否会发生再次购买行为，属于二分类问题。
模型采用 LightGBM，并以 AUC 作为主要评价指标。

二、目录结构说明
本项目目录结构如下：

tianchi_repeat_buy/
├── data/
│   ├── train_format1.csv        训练数据（含标签）
│   ├── test_format1.csv         测试数据（无标签）
│   ├── user_log_format1.csv     用户行为日志
│   └── user_info_format1.csv    用户画像信息
│
├── baseline_lgb.py              最终模型代码
├── submission.csv               预测结果文件
└── README.txt                   项目说明文件

三、特征工程说明
以用户对商家的历史行为强度为核心特征，统计以下行为次数：
- click_cnt：点击次数
- cart_cnt：加购次数
- buy_cnt：购买次数
- fav_cnt：收藏次数

特征按 (user_id, merchant_id) 进行聚合，缺失值统一填充为 0。
在实验过程中对时间特征与比例特征进行了探索，但验证集结果显示其泛化效果有限，最终未纳入模型。

四、模型说明
- 模型类型：LightGBM（二分类）
- 训练方式：训练集按 8:2 划分为训练集和验证集
- 评价指标：AUC

该模型在验证集上取得稳定表现，能够有效区分是否发生复购行为。

五、运行说明
1. 确保已安装 Python 3.8 及以上版本
2. 安装依赖库：
   pip install pandas lightgbm scikit-learn
3. 在项目根目录下运行：
   python baseline_lgb.py
4. 程序运行完成后将生成 submission.csv 文件

六、结果文件说明
submission.csv 文件包含三列：
- user_id
- merchant_id
- prob（预测为复购的概率）
