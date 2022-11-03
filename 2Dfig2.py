from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)




## Parameters
L = 10.0 # domain is (0,L)x(0,L)
# initial condition
t_i = 0.
y_i = L/2.
x_i = 0.1
s_i = 0
theta_i = 0
# target
el = .5
Xx = L-1.5
Xy = L/2.
# rates
k = 0.5
beta = 6.
alpha = 4.
q = 0.5
qv = array([q/2.,q/2.,(1-q)/2.,(1-q)/2.])
Qd = array([q/2.,q,q+(1.-q)/2.,1.])
vx = array([1.,-1.,0.,0.])
vy = array([0.,0.,1.,-1.])

prams = dict(L=L,el=el,k=k,beta=beta,alpha=alpha,x_i=x_i,y_i=y_i)
np.save('fig2_prams_sk',prams)
########## MC sim

## q
Ni = 10
Nj = 2
T1 = zeros((Ni,Nj))
Var1 = zeros((Ni,Nj))
qv1 = linspace(0.1,0.9,Ni)
Xyv1 = linspace(el,L/2,Nj)
        
mcflag = 1
NI = 10000
if mcflag==0:
    for i in arange(Ni):        
        for j in arange(Nj):
            q = qv1[i]
            Xy = Xyv1[j]
            Qd = array([q/2.,q,q+(1.-q)/2.,1.])
            sim = zeros((1,2))        
            sim = mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI)
            #sim = mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            T1[i,j] = sim[0,0]
            Var1[i,j] = sim[0,1]
    np.save('fig2_data_T1_sk', T1)
    np.save('fig2_data_Var1_sk',Var1)
else:
    T1 = np.load('fig2_data_T1_sk.npy')

    
## Xy
Ni = 3
Nj = 20
T2 = zeros((Ni,Nj))
Var2 = zeros((Ni,Nj))
qv2 = linspace(0.1,0.9,Ni)
Xyv2 = linspace(el,L/2,Nj)
mcflag = 1
NI = 5000
if mcflag==0:
    for i in arange(Ni):
        for j in arange(Nj):
            q = qv2[i]
            Xy = Xyv2[j]
            Qd = array([q/2.,q,q+(1.-q)/2.,1.])
            sim = zeros((1,2))        
            sim = mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI)
            #sim = mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            T2[i,j] = sim[0,0]
            Var2[i,j] = sim[0,1]
    np.save('fig2_data_T2_sk', T2)
    np.save('fig2_data_Var2_sk',Var2)   
else:
    T2 = np.load('fig2_data_T2_sk.npy')
    
def RegP(xi1,eta1,xi2,eta2,a1,a2):
    H0 = a1/3 + 1/(2*a1)*(eta1**2 + eta2**2) - max(eta1,eta2)
    z1 = exp(1j*pi*(xi1+xi2)/a2)
    z2 = exp(1j*pi*(xi1-xi2)/a2)
    zeta1 = exp(-pi/a2*abs(eta1+eta2))
    zeta2 = exp(-pi/a2*abs(eta1-eta2))
    sigma1 = exp(-pi/a2*(2*a1-abs(eta1+eta2)))
    sigma2 = exp(-pi/a2*(2*a1-abs(eta1-eta2)))
    if xi1==xi2 :
        arg1 =  pi/a2
    else:
        arg1 = abs(1-z2*zeta2)/sqrt((xi1-xi2)**2+(eta1-eta2)**2)
    arg2 = abs(1-z2*zeta1)*abs(1-z1*zeta2)*abs(1-z1*zeta1)*abs(1-z2*sigma2)*abs(1-z2*sigma1)*abs(1-z1*sigma2)*abs(1-z1*sigma1)
    tau = exp(-2*pi*a1/a2)
    hot = 0
    for j in arange(1,1):
        arghot1 = abs(1-tau**j*z2*zeta2)*abs(1-tau**j*z2*zeta1)*abs(1-tau**j*z1*zeta2)*abs(1-tau**j*z1*zeta1)
        arghot2 = abs(1-tau**j*z2*sigma2)*abs(1-tau**j*z2*sigma1)*abs(1-tau**j*z1*sigma2)*abs(1-tau**j*z1*sigma1)
        hot = hot + log(arghot1) + log(arghot2)
    R = 2*pi*H0/a2 - log(arg1) - log(arg2) - hot
    return R
def apprx(q,Xx,Xy,x_i,y_i):
    a1 = 1./sqrt(2*(1-q)) 
    a2 = 1./sqrt(2*q)
    xi0 = a2*x_i/L  # x_i is the initial value of x, but xi0 is the rescaled (greek xi) initial value of x
    eta0 = a1*y_i/L
    Xxi =  a2*Xx/L # Xxi referes to the rescaled (greek xi) target coordinate
    Xeta = a1*Xy/L
    a = alpha/(alpha+beta) 
    b = beta/(alpha+beta)
    lam0 = k*b  #  effective absorbtion rate
    D = a/(2*beta)  # effective diffusivity
    ### leading order QSS
    lam = lam0
    Dk = D
    ### higher order QSS
    lam = lam0 - a/beta*lam0**2  + a*(a-b)/beta**2*lam0**3
    Dk = D + k*a*b*(b+1.)/(2.*beta**2) +  (k*b)**2*a*(1-(a-b)*(b+1))/(2*beta**3)
    ### decoupling for T_0
    #lam = lam0
    #Dk = D*(alpha+k)/alpha
    ### decoupling for \bar{T}
    #lam = (alpha+beta)/(alpha+beta+k)*b*k
    #Dk = (alpha+k)/(beta+alpha+k)/(2*beta) # modified effective diffusivity
    tdist = sqrt((xi0-Xxi)**2 + (eta0-Xeta)**2)  # distance to target
    d1 = 1/2.*(a1+a2)  # logarithmic capacitance of the target
    d3 = sqrt(a1*a2)
    Rdif = RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)  -  RegP(xi0,eta0,Xxi,Xeta,a1,a2) # Regular part of the Green's function
    singdif = log(tdist) - log(el*d1/L)  ## singular part of the Green's function
    lDs = el*d3/sqrt(Dk/lam)  # target radius / absorbtion length scale
    tD = L**2/D       # diffusion time scale to explore the domain
    Tx = 1./lam + a1*a2*tD/(2*pi) *  special.iv(0,lDs)/special.iv(1,lDs)/lDs  # Time to target, starting at the target
    if isnan(Tx):
        Tx=0    
    return Tx + a1*a2*tD/(2*pi)*(singdif + Rdif)  # MFPT
vec_apprx = vectorize(apprx)

## q
[qg1,Xyg1] = meshgrid(linspace(qv1[0],qv1[-1]),linspace(Xyv1[0],Xyv1[-1],2))
Ta11 = vec_apprx(qg1,Xx,Xyg1,x_i,y_i)


## Xy
[qg2,Xyg2] = meshgrid(linspace(qv2[0],qv2[-1],3),linspace(Xyv2[0],Xyv2[-1]));
Ta21 = vec_apprx(qg2,Xx,Xyg2,x_i,y_i);


##### Figure
fig2 = figure(2,figsize=(10,5),facecolor='none',edgecolor='none')
clf()
figtext(0.02,0.9,'a',fontsize=20)
figtext(0.5,0.9,'b',fontsize=20)

ax1=fig2.add_subplot(121)
#plot(qg1.transpose(),Ta11.transpose()/60,lw=2)
plot(qg1[0,:],Ta11[0,:],'k',lw=2)
plot(qg1[0,:],Ta11[1,:],'--k',lw=2)
plot(qv1,T1[:,0],'+',mec='#996600',mew=2)
plot(qv1,T1[:,1],'3',mec='#809900',mew=2)
xlabel(r'$q$',fontsize=24)
ylabel(r'$T$',fontsize=24)
axis([0.09,0.91,2000,8000])
leg1 = ax1.legend((r'$y_0=l$',r'$y_0=L/2$'),loc='upper center')
for t in leg1.get_texts():
    t.set_fontsize(20)    # the legend text fontsize
leg1.get_frame().set_facecolor('none')
leg1.get_frame().set_edgecolor('none')


ax2=fig2.add_subplot(122)

#plot(Xyg2,Ta21/60,lw=2)
plot(Xyg2[:,0],Ta21[:,0],'k',lw=2)
plot(Xyg2[:,0],Ta21[:,1],'--k',lw=2)
plot(Xyg2[:,0],Ta21[:,2],'gray',lw=2)

#plot(Xyv2,T2.transpose()/60,'o')
plot(Xyv2,T2[0,:],'s',mfc='#A2E495',mew=0.5,ms=4)
plot(Xyv2,T2[1,:],'o',mfc='#B333CC',mew=0.5,ms=4)
plot(Xyv2,T2[2,:],'d',mfc='#996600',mew=0.5,ms=4)

xlabel(r'$y_0$',fontsize=24)
ylabel(r'$T$',fontsize=24)
leg2 = ax2.legend((r'$q=0.1$',r'$q=0.5$',r'$q=0.9$'))
for t in leg2.get_texts():
    t.set_fontsize(20)    # the legend text fontsize
leg2.get_frame().set_facecolor('none')
leg2.get_frame().set_edgecolor('none')
axis([el-0.01,L/2+0.01,2000,10000])
##savefig('fig2.pdf',format='pdf',transparent='true')
savefig('fig2.eps',format='eps')
