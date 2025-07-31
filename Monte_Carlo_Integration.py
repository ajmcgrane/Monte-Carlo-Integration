import numpy as np
import time

#function to be integrated
def f(x):
    return (np.sin(x))

#Monte Carlo Integration function
#integrates f(x) from a to b using N random samples
def monteCarloIntegration(a, b, N):
    #note: np.random.uniform(a, b, N) technically uses the interval [a, b), but P(X_i=b)=0 for X_i~U[a, b] anyway, so this is fine
    xrand = np.random.uniform(a, b, N)
    answer = (b - a) * np.sum(f(xrand)) / N
    
    return answer

#inputs
a = 0
b = 2 * np.pi
N = 100000

#print(monteCarloIntegration(a, b, N))

start = time.time()
numTrials = 1000
trials = np.zeros(numTrials)
for i in range(numTrials):
    trials[i] = monteCarloIntegration(a, b, N)
end = time.time()
print("Monte Carlo Integration took " + str(end - start) + " seconds")
print("mean: ")
print(np.mean(trials))
print("var: ")
print(np.var(trials))

