import numpy as np
from scipy import stats



# [https://towardsdatascience.com/how-to-use-python-to-figure-out-sample-sizes-for-your-study-871f0a76a19c](https://towardsdatascience.com/how-to-use-python-to-figure-out-sample-sizes-for-your-study-871f0a76a19c)
# https://towardsdatascience.com/power-analysis-made-easy-dfee1eb813a

ATS = [6, 5, 3.7, 3.5, 3]
slow = [1.1, 2.6, 3.5, 3.7, 3.7]
fast = [5.5, 5.4, 3.1, 2.9, 0.5]



# ATS VS SLOW
print("ATS vs SLOW")
p = (np.mean(ATS) + np.mean(slow))/(len(ATS)+len(slow))
print('p:', p)
se = np.sqrt(p*(1-p)*((1/len(ATS))+(1/len(slow))))
print('se:', se)

# The critical value at 0.05 significance is ~1.96 which you can obtain from a look-up table.

# If we now consider the sampling distribution given the alternative hypothesis, then we want the
# area under the curve between -1.96 and 1.96 to equal 20% (for 80% power). Therefore, the critical
# value has to be a distance of ~0.84 away from the mean (also available from a look-up table).

# So the total standardized difference between means must be 1.96 + 0.84 = 2.8.
dist = 2.8
delta = np.mean(ATS) - np.mean(slow)

n = (dist/delta)**2*p*(1-p)*2
print("n:", n)

# ATS VS FAST
print("ATS vs FAST")
p = (np.mean(ATS) + np.mean(fast))/(len(ATS)+len(fast))
print('p:', p)
se = np.sqrt(p*(1-p)*((1/len(ATS))+(1/len(fast))))
print('se:', se)
dist = 2.8
delta = np.mean(ATS) - np.mean(fast)

n = (dist/delta)**2*p*(1-p)*2
print("n:", n)