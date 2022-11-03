from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
from multiprocessing import Pool
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)



## Parameters
prams = np.load('fig4_prams_30_3_2011.npy').tolist()
#prams = np.load('fig4_prams_27_3_2011.npy').tolist()
#prams = np.load('fig4_prams_d.npy').tolist()
globals().update(prams)
## MC data
T=np.load('fig4_30_3_2011.npy')
#T=np.load('fig4_27_3_2011.npy')
#T=np.load('fig4_data_T1_d.npy')

def RegP(xi1,eta1,xi2,eta2,a1,a2):
    H0 = a1/3. + 1./(2.*a1)*(eta1**2 + eta2**2) - max(eta1,eta2);
    z1 = exp(1j*pi*(xi1+xi2)/a2);
    z2 = exp(1j*pi*(xi1-xi2)/a2);
    zeta1 = exp(-pi/a2*abs(eta1+eta2));
    zeta2 = exp(-pi/a2*abs(eta1-eta2));
    sigma1 = exp(-pi/a2*(2*a1-abs(eta1+eta2)));
    sigma2 = exp(-pi/a2*(2*a1-abs(eta1-eta2)));
    if xi1==xi2 :
        arg1 =  pi/(a2)
    else:
        arg1 = abs(1-z2*zeta2)/sqrt((xi1-xi2)**2+(eta1-eta2)**2)
    arg2 = abs(1-z2*zeta1)*abs(1-z1*zeta2)*abs(1-z1*zeta1)*abs(1-z2*sigma2)*abs(1-z2*sigma1)*abs(1-z1*sigma2)*abs(1-z1*sigma1);
    tau = exp(-2.*pi*a1/a2)
    hot = 0
    R = 2.*pi*H0/a2 - log(arg1) - log(arg2) + hot;
    return R;
def apprx(alpha,beta,method='QSS'):
    a1 = 1./sqrt(2.*(1.-q))
    a2 = 1./sqrt(2.*q)
    #xi0 = a2*x_i/L  # x_i is the initial value of x, but xi0 is the rescaled (greek xi) initial value of x
    #eta0 = a1*y_i/L
    Xxi =  a2*Xx/L # Xxi referes to the rescaled (greek xi) target coordinate
    Xeta = a1*Xy/L
    a = alpha/(alpha+beta)
    b = beta/(alpha+beta)
    kappa = alpha/(alpha+k)
    lam0 = b*k # leading order effective absorption rate    
    D = a/(2*beta)  # effective diffusivity
    ### leading order QSS
    #lam = lam0
    #Dk = D
    if method == 'QSS': ### higher order QSS        
        #lam = lam0 - a/beta*lam0**2  +  a*(a-b)/beta**2*lam0**3 - a*(a*b-(a-b)**2)/beta**3*lam0**4
        #Dk = D + k*a*b*(b+1.)/(2.*beta**2) + (k*b)**2*a*(1-(a-b)*(b+1))/(2*beta**3)
        NEPS = 10
        mu = zeros(NEPS)
        phi = zeros(NEPS)
        rat = zeros(NEPS)
        mu[0] = -a/beta*b**2
        #mu[1] = a/beta**2*(a-b)*b**3*k
        mu[1] = 1/beta*b*k*(b-a)*mu[0]
        #mu[2] =  -a/beta**3*(a*b-(a-b)**2)*b**4*k**2
        mu[2] = 1/beta*(b*k*(b-a)*mu[1] + b*k**2*mu[0]**2)
        phi[0] = -a/beta
        #phi[1] = -a/beta**2*(b+1)*b*k
        phi[1] = 1/beta*k*(b*phi[0] + mu[0])
        #phi[2] = -(k*b)**2*a*(1-(a-b)*(b+1))/beta**3
        phi[2] = 1/beta*(k*(b*phi[1] + mu[1]) + k**2*phi[0]*mu[0])
        for j in arange(3,NEPS):
            mu[j] = 1/beta*(b*k*(b-a)*mu[j-1] + b*k**2*sum(mu[0:j-2]*mu[0:j-2][-1::-1]))
            phi[j] = 1/beta*(k*(b*phi[j-1] + mu[j-1]) + k**2*sum(phi[0:j-2]*mu[0:j-2][-1::-1]))
        lam = lam0 + k**2*sum(mu)
        Dk =  - 1./2.*sum(phi)
        #print abs(lam/Dk-lam0/D*kappa)*D*kappa/lam0
        #print lam/Dk, lam0*kappa/D
    else: ### decoupling for T_0        
        lam = lam0
        Dk = D/kappa
    ### decoupling for \bar{T}
    #lam = (alpha+beta)/(alpha+beta+k)*b*k
    #Dk = (alpha+k)/(beta+alpha+k)/(2*beta) # modified effective diffusivity   
    #tdist = sqrt((xi0-Xxi)**2 + (eta0-Xeta)**2)  # distance to target
    #Rdif = RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)  -  RegP(xi0,eta0,Xxi,Xeta,a1,a2) # Regular part of the Green's function
    #singdif = log(tdist) - log(el*d/L)  ## singular part of the Green's function
    lDs = el/sqrt(Dk/lam)  # target radius / absorbtion length scale
    tD = L**2/D       # diffusion time scale to explore the domain
    Tx = 1/lam0 + 1/(2.*pi)*tD *  special.iv(0,lDs)/special.iv(1,lDs)/lDs  # Time to target, starting at the target
    #return Tx + a1*a2/(2*pi)*tD*(singdif + Rdif);  # MFPT
    return Tx + a1*a2/(2.*pi)*tD*(- log(el/L) + RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)) #GMFPT

vec_apprx = vectorize(apprx)
## analytical solutions
[alpha,beta] = meshgrid(linspace(alpha_lb,alpha_ub,Ni),linspace(beta_lb,beta_ub,Nj)) # for comparison to MC
[alpha2,beta2] = meshgrid(linspace(alpha_lb,alpha_ub,75),linspace(beta_lb,beta_ub,75)) # for plotting
Ta_qss_c = vec_apprx(alpha,beta)
Ta_qss_p = vec_apprx(alpha2,beta2)
Ta_dc_c = vec_apprx(alpha,beta,'DC')
Ta_dc_p = vec_apprx(alpha2,beta2,'DC')

## stuff for plotting
ar = (alpha_ub-alpha_lb)/(beta_ub-beta_lb)
Rxx =RegP(Xx/L,Xy/L,Xx/L,Xy/L,1./sqrt(2.*(1.-q)),1./sqrt(2.*q))

Tmin_dc = Ta_dc_p.min()
beta_dc = 1./el*(log(L/el)+Rxx)**(-0.5);
alpha_dc = sqrt(2*k/el)*(log(L/el)+Rxx)**(-0.25)
inds_dc = unravel_index(Ta_dc_p.argmin(),Ta_dc_p.shape)
alpha_dc_exact = alpha2[inds_dc]
beta_dc_exact = beta2[inds_dc]

Tmin_qss = Ta_qss_p.min()
inds_qss = unravel_index(Ta_qss_p.argmin(),Ta_qss_p.shape)
alpha_qss = alpha2[inds_qss]
beta_qss = beta2[inds_qss]

Tmin_mc = T.min()
inds_mc = unravel_index(T.transpose().argmin(),T.shape)
alpha_mc = alpha[inds_mc]
beta_mc = beta[inds_mc]

print sqrt((beta_mc-beta_qss)**2 + (alpha_mc-alpha_qss)**2)/sqrt(beta_mc**2+alpha_mc**2)
print sqrt((beta_mc-beta_dc)**2 + (alpha_mc-alpha_dc)**2)/sqrt(beta_mc**2+alpha_mc**2)

#Conts1 = array([1150,1200,1300,1500,2000])
#Conts2 = array([1100,1160,1200,1300,1500,2000])
Conts1 = array([675,700,750,800,900,1000,1100,1200])
Conts2 = array([675,700,750,800,900,1000,1100,1200])
fig2 = figure(2,figsize=(10,5),facecolor='none',edgecolor='none')
clf()
figtext(0.02,0.9,'a',fontsize=20)
figtext(0.34,0.9,'b',fontsize=20)
figtext(0.66,0.9,'c',fontsize=20)

ax1 = fig2.add_subplot(131)
CS2=contour(alpha2,beta2,Ta_qss_p,Conts2)
plt.clabel(CS2, inline=1, fontsize=10,fmt='%1.0f')
imshow(Ta_qss_p,origin='lower',cmap='YlOrRd',extent=(alpha_lb,alpha_ub,beta_lb,beta_ub),aspect = ar,vmin=Tmin_qss)
text(alpha_mc,beta_mc,'X',fontsize=20,ha='center',va='center')
text(alpha_qss,beta_qss,'O',fontsize=20,ha='center',va='center')
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$\beta$',fontsize=24)

ax2 = fig2.add_subplot(132)
CS2=contour(alpha2,beta2,Ta_dc_p,Conts2)
plt.clabel(CS2, inline=1, fontsize=10,fmt='%1.0f')
imshow(Ta_dc_p,origin='lower',cmap='YlOrRd',extent=(alpha_lb,alpha_ub,beta_lb,beta_ub),aspect = ar,vmin=Tmin_dc)
#text(alpha_dc_exact,beta_dc_exact,'O',fontsize=20,ha='center',va='center')
text(alpha_dc,beta_dc,'*',fontsize=20,ha='center',va='center')
text(alpha_mc,beta_mc,'X',fontsize=20,ha='center',va='center')
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$\beta$',fontsize=24)

ax3 = fig2.add_subplot(133)
CS=contour(alpha,beta,T.transpose(),Conts1)
plt.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
imshow((T).transpose(),origin='lower',cmap='YlGnBu',extent=(alpha_lb,alpha_ub,beta_lb,beta_ub),aspect = ar,vmin=Tmin_mc)
text(alpha_mc,beta_mc,'X',fontsize=20,ha='center',va='center')
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$\beta$',fontsize=24)
#savefig('fig3.pdf',format='pdf')


fig1 = figure(1)
clf()
figtext(0.02,0.9,'a',fontsize=20)
figtext(0.5,0.9,'b',fontsize=20)
fig1.add_subplot(121)
CS=contour(alpha,beta,T.transpose(),Conts1)
plt.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
pcolor(alpha,beta,abs((T.transpose()-Ta_qss_c)/T.transpose()),cmap='YlOrRd',vmin=0,vmax=0.1)
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$\beta$',fontsize=24)
colorbar()

fig1.add_subplot(122)
CS=contour(alpha,beta,T.transpose(),Conts1)
plt.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
pcolor(alpha,beta,abs((T.transpose()-Ta_dc_c)/T.transpose()),cmap='YlOrRd',vmin=0,vmax=0.1)
xlabel(r'$\alpha$',fontsize=24)
ylabel(r'$\beta$',fontsize=24)
colorbar()
#savefig('fig4b.pdf',format='pdf')


