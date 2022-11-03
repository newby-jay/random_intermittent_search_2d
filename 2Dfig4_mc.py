from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
from multiprocessing import Pool
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)




## Parameters
L = 10. # domain is (0,L)x(0,L)
# initial condition
t_i = 0.
y_i = L/2.
x_i = L/2.
s_i = 0 # 1 = searching and 0 = moving
theta_i = 0.0
# target
el = .5
Xx = L/2.
Xy = L/2.
# rates
k = 1


q = 0.5
qv = array([q/2.,q/2.,(1-q)/2.,(1-q)/2.])
Qd = array([q/2.,q,q+(1.-q)/2.,1.])
vx = array([1.,-1.,0.,0.])
vy = array([0.,0.,1.,-1.])


prams =  dict(L= L, Ni= 20, Nj= 20, Xx= L/2., Xy = L/2., alpha_lb = 1.0, alpha_ub = 4.0, beta_lb = 1.0, beta_ub = 4.0, el = el, k = k, q= 0.5)
np.save('fig4_prams_30_3_2011',prams)


########## MC sim

## alpha
Ni = 20
Nj = 20
T1 = zeros((Ni,Nj))
Var1 = zeros((Ni,Nj))
alphav1 = linspace(1.0,4.0,Ni)
betav1 = linspace(1.0,4.0,Nj)


NI = 10**5
def MCloop(beta):
    T = zeros((Ni,1))
    for i in arange(Ni):
       alpha = alphav1[i]    
       sim = zeros((1,2))        
       #sim = mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI)
       sim = mc_sim_rd_avg(t_i,s_i,L,el,Xx,Xy,1.,alpha,beta,k,NI)
       #sim = mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
       T[i] = sim[0,0] 
    #Var[i] = sim[0,1]
    return T

pool = Pool(2)
result = pool.map(MCloop, betav1.transpose())
for j in arange(Nj):
    T1[:,j] = result[j][:,0]
np.save('fig4_30_3_2011', T1)
pool.close()


