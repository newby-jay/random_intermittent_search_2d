from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
from multiprocessing import Pool




## Parameters
L = 15.0 # domain is (0,L)x(0,L)
# initial condition
t_i = 0.
y_i = 0.
x_i = 0.
s_i = 1

# rate
beta = 7.
alpha = 7.
q = 0.5
qv = array([q/2.,q/2.,(1-q)/2.,(1-q)/2.])
Qd = array([q/2.,q,q+(1.-q)/2.,1.])



########## MC sim
Nj = 20
vb = linspace(0.,30.,Nj)
T = zeros(vb.shape)

mcflag = 1
NI = 1000
def MCloop(vb):
   Tl = 0;
   L_hat = L*(1.+vb);
   a = alpha/(alpha+beta)
   vsy = sqrt(1. + vb + 1./2.*vb**2 - a*(2.-a)*vb**2/8. ) 
   vx = array([1.+vb,-1.,0.,0.])
   vy = array([0.,0.,vsy,-vsy])
   vdx = a*vb/4.
   vdy = 0
  
   sim = zeros((1,2))        
   sim = mc_sim_5s_bias(t_i,x_i,y_i,s_i,L_hat,Qd,vx,vy,vdx,vdy,alpha,beta,NI)
   Tl = sim[0,0]
   #Var[i] = sim[0,1]  
   return Tl
if mcflag == 0:
    pool = Pool(1)
    result = pool.map(MCloop,vb)
    for j in arange(Nj):
        #T[j] = MCloop(vb[j])        
        T[j] = result[j]
    np.save('fig5_data_T_a', T)
    pool.close()
    #np.save('fig5_data_Var1',Var1)
else:
    T = np.load('fig5_data_T_a.npy')


## analytical
vba = linspace(vb[0],vb[-1],Nj)
a = alpha/(alpha+beta)
b = 1-a
Dxx = a/(2.*beta)*(1. + vba + 1./2.*vba**2);
Dxx_hat = a/(2.*beta)*(1. + vba + 3./8.*vba**2) +  a/(2*beta)*b**2*vba**2/8.;
Ta_hat = L**2*(1.+vba)**2/(4*Dxx_hat);
Ta = L**2*(1.+vba)**2/(4*Dxx);
##### Figure
fig1 = figure(1,figsize=(10,5),facecolor='none',edgecolor='none')
clf()
## figtext(0.03,0.9,'a',fontsize=20)
## figtext(0.5,0.9,'b',fontsize=20)

#plot(vb,T,'*')
#plot(vba,Ta_hat,'k',vba,Ta,':k')
plot(vb, abs((T-Ta_hat)/T),'k',vb,abs((T-Ta)/T),':k')

## ax1 = fig1.add_subplot(121)

xlabel(r'$v_{\mathrm{bias}}$',fontsize=24)
ylabel(r'$T$',fontsize=24)
## leg1 = ax1.legend(loc='upper left')
## for t in leg1.get_texts():
##     t.set_fontsize(20)    # the legend text fontsize
## leg1.get_frame().set_facecolor('none')
## leg1.get_frame().set_edgecolor('none')
## ax1.get_frame().set_facecolor('none')

savefig('fig5.eps',format='eps')


s=0
for j in arange(1,100):
   for k in arange(1,100):
      s = s+ sin(j*pi/2.)*sin(k*pi/2.)/(j**2 + k**2)

