# 说明

- 所有参数设置均在<code>main.py</code>的<code>args</code>中
- <code>vec_len</code>是词向量的长度
- <code>max_l</code>是每条评论词数量，为了节省训练时间，实验截取每条评论的前100词，不足补0向量
- 采用3、4、5三个宽度的卷积核，每个核输出channel为100
- 梯度算法采用ADADELTA
- 运行会时会保存生成的数据划分集，并保存在dev set上表现最好的model
- <code>eval</code>参数为真时测试all_model在各数据集上的表现

# 运行

- python版本：3.6.5

使用命令
```
python main "books"
```
将<code>books</code>换为各数据集名称


