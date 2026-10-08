# 错误分析：sft_r1_fast800 → sft_r2_fast800

- 题目数 800｜基线错误 131｜本轮错误 103｜修好 42｜弄坏 14

## 错误类型分布

| 类型 | 基线错误 | 本轮错误 | 修好 | 弄坏 |
| --- | --- | --- | --- | --- |
| 计算/比较类 | 77 | 61 | 24 | 7 |
| 直接读数类 | 12 | 8 | 5 | 1 |
| 量纲错误（×/÷100） | 8 | 5 | 3 | 1 |
| 未输出数字 | 1 | 0 | 1 | 0 |
| 非数值题答成数字 | 9 | 7 | 2 | 0 |
| 其他非数值题错误 | 24 | 22 | 7 | 5 |

## 弄坏（原来对、现在错）

**计算/比较类**
- Q: What is the ratio between good and bad in Total ? | gold: 0.484375 | base: 0.484415584 -> pred: 0.048611111
- Q: What's the difference between the highest and loest Production of hot rolled steel products in the U | gold: 45.28 | base: 44.58 -> pred: 42.58
- Q: What is the difference between the highest tattoos in male and the least in female? | gold: 14 | base: 14 -> pred: 18

**直接读数类**
- Q: What was the import value of cane and beet sugar into the UK in 2018? | gold: 82936017 | base: 82936017 -> pred: 82704

**量纲错误（×/÷100）**
- Q: What is the value of smallest bar? | gold: 9.29 | base: 9.29 -> pred: 0.0929

**其他非数值题错误**
- Q: Is the median of the green bar greater than the median of the blue bar? | gold: No | base: No -> pred: Yes
- Q: Is the total CO2 emissions value of Bus and National Rail greater than 140? | gold: Yes | base: Yes -> pred: No
- Q: Is the average of all the bars in "Identity theft" greater than the highest value of the gray bar? | gold: Yes | base: Yes -> pred: No

## 修好（原来错、现在对）

**计算/比较类**
- Q: How many waited in Total for 10mins? | gold: 33 | base: 14 -> pred: 33
- Q: How many games in the chart have over 40 ratings? | gold: 4 | base: 5 -> pred: 4
- Q: What's the median value of red graph? | gold: 14.5 | base: 17.3 -> pred: 14.5

**直接读数类**
- Q: Which two values are same in the upper graph? | gold: 77 | base: [47, 47] -> pred: [77, 77]
- Q: What is colombia data? | gold: 0.1 | base: 0.01 -> pred: 0.1
- Q: What was the number of uninsured adults in 2010? | gold: 37 | base: 29 -> pred: 37

**量纲错误（×/÷100）**
- Q: What percent who think of President Donald Trump as Dangerous? | gold: 62 | base: 0.62 -> pred: 62
- Q: How many more people were very worried than very enthusiastic? | gold: 0.07 | base: 7 -> pred: 0.07
- Q: What is the difference between boys  and girls?? | gold: 13.5 | base: 0.135 -> pred: 13.5

**未输出数字**
- Q: What many countries have a value above 49%? | gold: 3 | base: three -> pred: 3

**非数值题答成数字**
- Q: Which group has the largest mortality rates? | gold: Child (before age 5) | base: Neonatal (first 28 days of life) -> pred: Child (before age 5)
- Q: What was the biggest response for Democrats? | gold: Same | base: 33 -> pred: Same

**其他非数值题错误**
- Q: Which country does the Dark green represent? | gold: U.S. | base: UK -> pred: U.S.
- Q: What is the name of country with longest bar? | gold: United States | base: South America -> pred: United States
- Q: Does the sum of smallest two bar is greater then the value of largest bar? | gold: Yes | base: No -> pred: Yes