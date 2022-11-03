from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)




## Parameters
L = 10.0; # domain is (0,L)x(0,L)
# initial condition
t_i = 0.;
y_i = L/2.;
x_i = 0.1;
s_i = 0;
theta_i = 0;
# target
el = .5;
Xx = L/2;
Xy = L/2.;
# rates
k = 0.5;
beta = 4.;
alpha = 2.;
q = 0.5;
qv = array([q/2.,q/2.,(1-q)/2.,(1-q)/2.]);
Qd = array([q/2.,q,q+(1.-q)/2.,1.]);
vx = array([1.,-1.,0.,0.]);
vy = array([0.,0.,1.,-1.]);


########## MC sim

## alpha
Ni = 20
Nj = 2
T1 = zeros((Ni,Nj))
STD1 = zeros((Ni,Nj))
alphav1 = linspace(1.0,5.0,Ni)
betav1 = linspace(1.0,5.0,Nj)
        
mcflag = 1
NI = 5000
if mcflag==0:
    for j in arange(Nj):
        beta = betav1[j]
        for i in arange(Ni):
            alpha = alphav1[i]                                
            sim = zeros((1,2))        
            #sim = mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI)
            #sim = mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            sim = mc_sim_rd_avg(t_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            T1[i,j] = sim[0,0]
            STD1[i,j] = sim[0,1]
    np.save('fig1_data_T1', T1)
    np.save('fig1_data_STD1',STD1)
else:
    T1 = np.load('fig1_data_T1.npy')
    STD1 = np.load('fig1_data_STD1.npy')
    
## Beta
Ni = 2;
Nj = 20;
T2 = zeros((Ni,Nj));
STD2 = zeros((Ni,Nj));
alphav2 = linspace(1.0,5.0,Ni);
betav2 = linspace(1.0,5.0,Nj);
mcflag = 1
NI = 5000
if mcflag==0:
    for i in arange(Ni):
        alpha = alphav2[i]
        for j in arange(Nj):
            beta = betav2[j]                    
            sim = zeros((1,2))        
            #sim = mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI)
            #sim = mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            sim = mc_sim_rd_avg(t_i,s_i,L,el,Xx,Xy,1,alpha,beta,k,NI)
            T2[i,j] = sim[0,0]
            STD2[i,j] = sim[0,1]
    np.save('fig1_data_T2', T2)
    np.save('fig1_data_STD2',STD2)   
else:
    T2 = np.load('fig1_data_T2.npy')
    STD2 = np.load('fig1_data_STD2.npy')


def RegP(xi1,eta1,xi2,eta2,a1,a2):
    H0 = a1/3 + 1/(2*a1)*(eta1**2 + eta2**2) - max(eta1,eta2);
    z1 = exp(1j*pi*(xi1+xi2)/a2);
    z2 = exp(1j*pi*(xi1-xi2)/a2);
    zeta1 = exp(-pi/a2*abs(eta1+eta2));
    zeta2 = exp(-pi/a2*abs(eta1-eta2));
    sigma1 = exp(-pi/a2*(2*a1-abs(eta1+eta2)));
    sigma2 = exp(-pi/a2*(2*a1-abs(eta1-eta2)));
    if xi1==xi2 :
        arg1 =  pi/a2
    else:
        arg1 = abs(1-z2*zeta2)/sqrt((xi1-xi2)**2+(eta1-eta2)**2)
    arg2 = abs(1-z2*sigma2)*abs(1-z2*sigma1)*abs(1-z1*sigma2)*abs(1-z1*sigma1)*abs(1-z2*zeta1)*abs(1-z1*zeta2)*abs(1-z1*zeta1);
    tau = exp(-2*pi*a1/a2)
    hot = 0
    for j in arange(1,0):
        arghot1 = abs(1-tau**j*z2*zeta2)*abs(1-tau**j*z2*zeta1)*abs(1-tau**j*z1*zeta2)*abs(1-tau**j*z1*zeta1)
        arghot2 = abs(1-tau**j*z2*sigma2)*abs(1-tau**j*z2*sigma1)*abs(1-tau**j*z1*sigma2)*abs(1-tau**j*z1*sigma1)
        hot = hot - log(arghot1) - log(arghot2)
    R = 2.*pi*H0/a2 - log(arg1) - log(arg2) + hot;
    return R;
def apprx(alpha,beta):
    a1 = 1./sqrt(2*(1-q));
    a2 = 1./sqrt(2*q);
    xi0 = a2*x_i/L;  # xi is the initial value of x, but xi0 is the rescaled (greek xi) initial value of x
    eta0 = a1*y_i/L;
    Xxi =  a2*Xx/L; # Xxi referes to the rescaled (greek xi) target coordinate
    Xeta = a1*Xy/L;
    a = alpha/(alpha+beta)
    b = beta/(alpha+beta)
    kappa = alpha/(alpha+k)
    lam0 = b*k # leading order effective absorption rate    
    D = a/(2*beta)  # effective diffusivity
    
    ### leading order QSS
    #lam = lam0
    #Dk = D
    ### higher order QSS
    #lam = lam0 - a/beta*lam0**2  +  a*(a-b)/beta**2*lam0**3 - a*(a*b-(a-b)**2)/beta**3*lam0**4
    #Dk = D + k*a*b*(b+1.)/(2.*beta**2) + (k*b)**2*a*(1-(a-b)*(b+1))/(2*beta**3)
    NEPS = 5
    mu = zeros(NEPS)
    phi = zeros(NEPS)
    rat = zeros(NEPS)
    mu[0] = -a/beta*b**2
    mu[1] = a/beta**2*(a-b)*b**3*k
    mu[2] =  -a*(a*b-(a-b)**2)/beta**3*b**4*k**2
    phi[0] = -a/beta
    phi[1] = -a/beta**2*(b+1)*b*k        
    phi[2] = -(k*b)**2*a*(1-(a-b)*(b+1))/beta**3        
    for j in arange(3,NEPS):
        mu[j] = 1/beta*(b*k*(b-a)*mu[j-1] + b*k**2*sum(mu[0:j-2]*mu[0:j-2][-1::-1]))
        phi[j] = 1/beta*(k*(b*phi[j-1] + mu[j-1]) + k**2*sum(phi[0:j-2]*mu[0:j-2][-1::-1]))
    lam = lam0 + k**2*sum(mu)
    Dk =  - 1./2.*sum(phi)
    ### decoupling for T_0
    #lam = lam0
    #Dk = D/kappa
    ### decoupling for \bar{T}
    #lam = (alpha+beta)/(alpha+beta+k)*b*k
    #Dk = (alpha+k)/(beta+alpha+k)/(2*beta) 
    
    tdist = sqrt((xi0-Xxi)**2 + (eta0-Xeta)**2);  # distance to target
    d = 1/2.*(a1+a2);  # logarithmic capacitance of the target
    epsi = d*el/L;
    Rdif = RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)  -  RegP(xi0,eta0,Xxi,Xeta,a1,a2); # Regular part of the Green's function
    singdif = log(tdist) - log(el*d/L);  ## singular part of the Green's function
    lDs = el*d/sqrt(Dk/lam);  # target radius / absorbtion length scale
    tD = L**2/D       # diffusion time scale to explore the domain
    Tx = 1./lam0 + a1*a2/(2*pi)*tD*  special.iv(0,lDs)/special.iv(1,lDs)/lDs;  # Time to target, starting at the target
    #return Tx + a1*a2/(2*pi)*tD*(singdif + Rdif);  # MFPT
    return Tx + a1*a2/(2*pi)*tD*(- log(el*d/L) + RegP(Xxi,Xeta,Xxi,Xeta,a1,a2))
vec_apprx = vectorize(apprx)

## alpha
[betag1,alphag1] = meshgrid(linspace(betav1[0],betav1[-1],2),linspace(alphav1[0],alphav1[-1],20));
Ta1 = vec_apprx(alphag1,betag1);
## beta
[betag2,alphag2] = meshgrid(linspace(betav2[0],betav2[-1],20),linspace(alphav2[0],alphav2[-1],2));
Ta2 = vec_apprx(alphag2,betag2);


##### Figure
fig1 = figure(1,figsize=(10,5),facecolor='none',edgecolor='none')
clf()
figtext(0.03,0.9,'a',fontsize=20)
figtext(0.5,0.9,'b',fontsize=20)

ax1 = fig1.add_subplot(121)
2>3plot(alphag1[:,0],Ta1[:,0],'k',lw=2,label=r'$\beta=1$')
plot(alphag1[:,1],Ta1[:,1],'--k',lw=2,label=r'$\beta=5$')

plot(alphav1,T1[:,0],'+',mec='#996600',mew=2)
#errorbar(alphav1,T1[:,0],yerr = STD1[:,0])
plot(alphav1,T1[:,1],'3',mec='#809900',mew=2)
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$T_{\mathrm{av}}$',fontsize=24)
leg1 = ax1.legend(loc='upper center')
for t in leg1.get_texts():
    t.set_fontsize(20)    # the legend text fontsize
leg1.get_frame().set_facecolor('none')
leg1.get_frame().set_edgecolor('none')
ax1.get_frame().set_facecolor('none')
axis([1,5,500,2500])

ax2 = fig1.add_subplot(122)
plot(betag2[0,:],Ta2[0,:],'k',lw=2,label=r'$\alpha=1$')
plot(betag2[1,:],Ta2[1,:],'--k',lw=2,label=r'$\alpha=5$')
plot(betav2,T2[0,:],'s',mfc='#A2E495',mew=0.5,ms=4)
plot(betav2,T2[1,:],'o',mfc='#B333CC',mew=0.5,ms=4)
xlabel(r'$\beta$',fontsize=24)
ylabel(r'$T_{\mathrm{av}}$',fontsize=24)
leg2 = ax2.legend(loc='upper center')
for t in leg2.get_texts():
    t.set_fontsize(20)    # the legend text fontsize
leg2.get_frame().set_facecolor('none')
leg2.get_frame().set_edgecolor('none')
ax2.get_frame().set_facecolor('none')
axis([1,5,500,2500])

##savefig('fig1.pdf',format='pdf',transparent='true')
#savefig('fig1.eps',format='eps')


figure(2)
clf()
plot(abs((Ta1-T1)/T1))
plot(abs((Ta2-T2)/T2).transpose())
