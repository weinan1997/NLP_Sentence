# 运行

- python版本：3.6.5
- pytorch版本：0.4.0

# 命令行参数

-  `-m --model` 模型选项，有以下几种：cnn、gru、gru_random、gru_domain、gs，默认值为cnn
- `--max_l` 保留评论的长度，默认值为90，即保留数据集中前90%评论的长度
- `-d --data_set` 数据集，有books、dvd、electronics、kitchen、all，默认为books
- `-g --gpu` 指定GPU（并未正确生效，建议使用CUDA_VISIBLE_DEVICES来指定）
- `-s --seed` 设定随机种子
- `-c --cross_validation` 设定交叉验证选用哪一个划分作为测试集
- `--run_cv` 一次性跑完五折交叉验证
- `--test_all` 在所有数据上训练模型，完成后在各个领域的测试集上测试，返回测试结果（当此参数为`True`时`-d`将被自动改为all
- `-o --optim_func` 优化函数选择，目前有：SGD、Adam、Adagrad，默认为SGD
- `-l --learning_rate` 学习率，默认值为0.05
- `-p --print_attention` 输出attention的信息
- `-b --batch_size` 默认值为50
- `-e --epoch_num` 默认值为30

# 数据路径

- 本地数据路径：将sorted_data文件夹放在工程目录下
- 服务器数据路径为`../../../data1/weinan/`，默认工程目录位置在`~/`
- 数据预处理会生成`revs_W_map.matrix`，若已存在此文件则直接使用，不再进行数据预处理。此文件的生成与随机种子有关，建议更改随机种子后移除此文件重新生成

# 运行

训练模型：

```
CUDA_VISIBLE_DEVICES=2 python main.py -m gru_random --test_all True --run_cv True -o Adam -l 0.0001 -b 256 -e 30
```
这是训练模型的参数设置，最终结果见wiki的baselines

```
CUDA_VISIBLE_DEVICES=2 python main.py -m gru_random -p True -d books
```
这是查看attention信息的命令，在模型训练好后使用，目前只实现了gru_random的attention查看

