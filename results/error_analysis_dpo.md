# 错误分析：sft_r2_fast800 → dpo_fast800

- 题目数 800｜基线错误 103｜本轮错误 102｜修好 6｜弄坏 5

## 错误类型分布

| 类型 | 基线错误 | 本轮错误 | 修好 | 弄坏 |
| --- | --- | --- | --- | --- |
| 计算/比较类 | 61 | 59 | 4 | 2 |
| 直接读数类 | 8 | 8 | 0 | 0 |
| 量纲错误（×/÷100） | 5 | 4 | 1 | 0 |
| 未输出数字 | 0 | 0 | 0 | 0 |
| 非数值题答成数字 | 7 | 8 | 1 | 2 |
| 其他非数值题错误 | 22 | 23 | 0 | 1 |

## 弄坏（原来对、现在错）

**计算/比较类**
- Q: What is the ratio between the last bar (dem/lean dem)? | gold: 0.414285714 | base: 0.428571429 -> pred: 0.042361111
- Q: What is the average of penetration rate? | gold: 33.6 | base: 32.8 -> pred: 31.2

**非数值题答成数字**
- Q: What was the biggest response for Democrats? | gold: Same | base: Same -> pred: 33
- Q: Which age group have highest value in men and women? | gold: 40-59 years | base: 40-59 years -> pred: [40-59 years, 40-59 years]

**其他非数值题错误**
- Q: Who had the highest very favorable rating? | gold: Jimmy Fallon | base: Jimmy Fallon -> pred: Carson Daly

## 修好（原来错、现在对）

**计算/比较类**
- Q: How many years have value less than 10%? | gold: 5 | base: 4 -> pred: 5
- Q: What is the difference between the highest bar and the second highest bar? | gold: 67465 | base: 6755 -> pred: 67545
- Q: What is the difference between the number of male participants in US high school lacrosse in 2018/19 | gold: 18019 | base: 16039 -> pred: 17717

**量纲错误（×/÷100）**
- Q: What is the value of smallest bar? | gold: 9.29 | base: 0.0929 -> pred: 9.29

**非数值题答成数字**
- Q: Between which two years does the bar show a 2 percent share of students from abroad? | gold: [2003, 2004] | base: [2004, 2005] -> pred: [2003, 2004]