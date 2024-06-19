import numpy as np
from datetime import datetime 


precision = 1e-4
tests_per_iter = 8
"""
Comete polarization function: Here "for..." means "for xi, pi in zip(x, weights)"
(min_{y} sum(pi*alpha abs(xi-y)beta for...) / sum(pialpha for...)*(1/beta)
"""

def pol(x, weights, alpha, beta):

    def pol_aux(y: float):
        return np.sum((weights ** alpha) * (np.abs(x - y) ** beta))
    
    low = np.min(x)
    high = np.max(x)
    while high - low > precision:
        lin = np.linspace(low, high, tests_per_iter)
        f_lin = [pol_aux(y) for y in lin]
        i: int = np.argmin(f_lin)
        low = lin[max(0, i - 2)]
        high = lin[min(len(lin) - 1, i + 2)]
    y_star = (low + high) / 2

    return y_star, pol_aux(y_star)

def Comete_factory(alpha: float, beta: float):
    def metric_comete(x, weights):
      _, polarization = pol(x, weights, alpha, beta)
      return polarization
    return metric_comete

def Comete_minpoint_factory(alpha: float, beta: float):
  def metric_minpoint_comete(x, weights):
    y_star, _ = pol(x, weights, alpha, beta)
    return y_star
  return metric_minpoint_comete


def generate_permutations(n=5,ymin=0,ymax=100,total_sum=100):
    perms = []
    def generate_permutation(perm, n):
       if sum(perm) == total_sum and n == 0:
          if len(perms) % 1000000 == 0:
             print(f"[{datetime.now().time()}] {len(perms)} hists generated")
          return perms.append([a for a in perm])
       if  sum(perm) > total_sum or n == 0:
          return
       
       for i in range(ymin, ymax+1-sum(perm)):
          generate_permutation([i]+perm, n-1)
    
    generate_permutation([], n)
    return perms

def generate_pols(alpha, beta, x, ymin, ymax, total_sum):
    hists = generate_permutations(n=len(x),ymin=ymin,ymax=ymax,total_sum=total_sum)
    print(f"[{datetime.now().time()}] {len(hists)} total hists generated")
    comete = Comete_factory(alpha=alpha, beta=beta)

    pols = []
    for i, hist in enumerate(hists):
       pols.append(comete(x, np.asarray(hist)))
       if i % 500000 == 0:
          print(f"[{datetime.now().time()}] {i}-th polarization computed")

    print(f"[{datetime.now().time()}] all polarization computed")
    pols = list(set(pols))
    pols.sort()
    print(f"[{datetime.now().time()}] filtered and sorted")
    i = 0
    while i < len(pols) - 1:
        if np.isclose(pols[i], pols[i+1]):
            pols.pop(i+1)
        else:
            i += 1
    return pols


x = [0.0, 0.25, 0.5, 0.75, 1.0]
ymin = 0
ymax = 100
total_sum = 100
folder = "pop100"

for alpha, beta in [(1.2,1), (2,1)]:
   print(f"[{datetime.now().time()}] started alpha:{alpha},beta:{beta},x:{x},ymin:{ymin},ymax:{ymax},pop:{total_sum}")
   pols = generate_pols(alpha, beta, x, ymin, ymax, total_sum)
   file_path= f"{folder}/alpha{alpha}_beta{beta}"
   with open(file_path, 'w') as file:
        file.write(f"alpha:{alpha},beta:{beta},x:{x},ymin:{ymin},ymax:{ymax},pop:{total_sum}\n")
        file.write('\n'.join([str(p) for p in pols]))
   print(f"[{datetime.now().time()}] done")