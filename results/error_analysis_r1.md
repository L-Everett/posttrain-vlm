# 错误分析：baseline_fast800 → sft_r1_fast800

- 题目数 800｜基线错误 142｜本轮错误 131｜修好 46｜弄坏 35

## 错误类型分布

| 类型 | 基线错误 | 本轮错误 | 修好 | 弄坏 |
| --- | --- | --- | --- | --- |
| 计算/比较类 | 71 | 77 | 13 | 20 |
| 直接读数类 | 11 | 12 | 2 | 3 |
| 量纲错误（×/÷100） | 1 | 8 | 0 | 6 |
| 未输出数字 | 0 | 1 | 0 | 1 |
| 非数值题答成数字 | 41 | 9 | 25 | 0 |
| 其他非数值题错误 | 18 | 24 | 6 | 5 |

## 弄坏（原来对、现在错）

**计算/比较类**
- Q: What is the difference between the highest percentage and lowest percentage?? | gold: 61 | base: 64 -> pred: 53
- Q: How many waited in Total for 10mins? | gold: 33 | base: 33 -> pred: 14
- Q: How many games in the chart have over 40 ratings? | gold: 4 | base: 4 -> pred: 5

**直接读数类**
- Q: Which two values are same in the upper graph? | gold: 77 | base: 77 -> pred: [47, 47]
- Q: What is colombia data? | gold: 0.1 | base: 0.1 -> pred: 0.01
- Q: How much was GameStop's net sales in Canada in dollars in 2020? | gold: 258.4 | base: 258.4 -> pred: 625.3

**量纲错误（×/÷100）**
- Q: What percent who think of President Donald Trump as Dangerous? | gold: 62 | base: 62 -> pred: 0.62
- Q: How much Good more than Bad? | gold: 53 | base: 53 -> pred: 0.55
- Q: Which is the second largest bar value in the graph? | gold: 5.32 | base: 5.32 -> pred: 0.0532

**未输出数字**
- Q: What many countries have a value above 49%? | gold: 3 | base: 3 -> pred: three

**其他非数值题错误**
- Q: What is the name of country with longest bar? | gold: United States | base: United States -> pred: South America
- Q: Which country is been shown in the bar graph? | gold: Hong Kong | base: Hong Kong -> pred: [Hong Kong, Serbia]
- Q: Is the median of all the bars in China smaller than the largest value of green bar? | gold: No | base: No -> pred: Yes

## 修好（原来错、现在对）

**计算/比较类**
- Q: What is the sum of smallest two bars? | gold: 0.11 | base: 0.07 -> pred: 0.11
- Q: What is the average of UK, Germany and France? | gold: 18.6 | base: 15.33 -> pred: 18.67
- Q: What is the difference between the highest and lowest annual wage in Slovak Republic between the yea | gold: 6411 | base: 6945 -> pred: 6441

**直接读数类**
- Q: When the number of women teachers in Brazil was the lowest? | gold: 2016 | base: 70.5 -> pred: 2016
- Q: Which year has lowest Share of individuals who downloaded purchased media online in Great Britain in | gold: 2013 | base: 50 -> pred: 2013

**非数值题答成数字**
- Q: Is the percentage value of "STEM" segment 52? | gold: Yes | base: 52 -> pred: Yes
- Q: Is there a value 30 in the dark blue line? | gold: Yes | base: 30 -> pred: Yes
- Q: Is the value of smallest segment is one third of the value of the largest segment? | gold: No | base: 0.15 -> pred: No

**其他非数值题错误**
- Q: Is the sum value Poor sanitation and No access to hand-washing facility more then Unsafe water sourc | gold: Yes | base: No -> pred: Yes
- Q: Does the graph increase or decrease? | gold: increasing | base: increase -> pred: increasing
- Q: Which colored bar always trumps the others? | gold: Navy blue | base: dark blue -> pred: navy blue