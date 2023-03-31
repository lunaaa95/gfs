# config.yml
## 基本信息
### 作用
配置文件

### 前置条件
无

## 配置参数
### 网络代理
+ proxy: 连接到ChatGPT的网络代理，如http://localhost:7890

### OpenAI设置
+ api-key: 调用OpenAI GPT所需要的API key
+ model: 调用OpenAI哪一种模型
+ temperature: 期望回答的多样性，建议0~0.3

### json接口
+ time-series: 存放股价、交易量的json
+ news: 存放新闻的json
+ result: 存放结果的json


# generate.py
## 基本信息
### 作用
结合股价、交易量和新闻预测趋势、波动性和交易活跃程度

### 前置条件
config.yml

## 调用方法
python generate.py

## 参数列表
无

## 输出
config.yml中规定的result文件


# json接口
## time-series对应的json文件
+ company: 公司代码
+ date: 日期
+ open: 开盘价序列
+ close: 收盘价序列
+ high: 最高价序列
+ low: 最低价序列
+ share: 成交量序列
+ turnover: 换手率序列

***序列为日频数据,至少要一个月长**

## news对应的json文件
+ company: 公司代码
+ date: 日期
+ title： 新闻标题
+ body: 新闻正文

## result对应的json文件
+ raw_output: ChatGPT输出的全部回答
+ prediction: 从回答中提取的结论
    + trend: 预测的趋势
    + volatility: 预测的波动性
    + active_level: 预测的交易活跃程度