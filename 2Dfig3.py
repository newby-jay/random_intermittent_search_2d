from pylab import *
from IPython import genutils
from scipy import special
from mc_sim import *
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)




## Parameters
L = 5.0 # domain is (0,L)x(0,L)
# initial condition
t_i = 0.
y_i = L/2
x_i = .1
s_i = 0
theta_i = 0
# target
el = .25
Xx = L-1
Xy = L/2.
# rates
k = 100000
beta = 6.
alpha = 4.
q = 0.5
qv = array([q/2.,q/2.,(1-q)/2.,(1-q)/2.])
Qd = array([q/2.,q,q+(1.-q)/2.,1.])
vx = array([1.,-1.,0.,0.])
vy = array([0.,0.,1.,-1.])
def RegP(xi1,eta1,xi2,eta2,a1,a2):  
    H0 = a1/3 + 1/(2*a1)*(eta1**2 + eta2**2) - max(eta1,eta2)
    z1 = exp(1j*pi*(xi1+xi2)/a2)
    z2 = exp(1j*pi*(xi1-xi2)/a2)
    zeta1 = exp(-pi/a2*abs(eta1+eta2))
    zeta2 = exp(-pi/a2*abs(eta1-eta2))
    sigma1 = exp(-pi/a2*(2*a1-abs(eta1+eta2)))
    sigma2 = exp(-pi/a2*(2*a1-abs(eta1-eta2)))
    if xi1==xi2 :
        arg1 =  pi/(a2)    
    else:
        arg1 = abs(1-z2*zeta2)/sqrt((xi1-xi2)**2+(eta1-eta2)**2)
    arg2 = abs(1-z2*zeta1)*abs(1-z1*zeta2)*abs(1-z1*zeta1)*abs(1-z2*sigma2)*abs(1-z2*sigma1)*abs(1-z1*sigma2)*abs(1-z1*sigma1)
    tau = exp(-2*pi*a1/a2)
    hot = 0
    for j in arange(1,2):
        arghot1 = abs(1-tau**j*z2*zeta2)*abs(1-tau**j*z2*zeta1)*abs(1-tau**j*z1*zeta2)*abs(1-tau**j*z1*zeta1)
        arghot2 = abs(1-tau**j*z2*sigma2)*abs(1-tau**j*z2*sigma1)*abs(1-tau**j*z1*sigma2)*abs(1-tau**j*z1*sigma1)
        hot = hot - log(arghot1) - log(arghot2)
    R = 2.*pi*H0/a2 - log(arg1) - log(arg2) + hot
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
    lam = alpha/(alpha+k)*k*b  #  effective absorbtion rate
    D = a/(2.*beta)  # effective diffusivity
    tdist = sqrt((xi0-Xxi)**2 + (eta0-Xeta)**2)  # distance to target
    d1 = 1/2.*(a1+a2)  # logarithmic capacitance of the target
    d2 = 2*a1/pi*special.ellipe(sqrt(1-(a2/a1)**2))
    d3 = sqrt(a1*a2)
    Rdif = RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)  -  RegP(xi0,eta0,Xxi,Xeta,a1,a2) # Regular part of the Green's function
    singdif = log(tdist) - log(el*d1/L)  ## singular part of the Green's function
    lDs = el*d3/sqrt(D/lam)  # target radius / absorbtion length scale
    tD = L**2/D       # diffusion time scale to explore the domain
    Tx = 1./lam + a1*a2/(2*pi)*tD *  special.iv(0,lDs)/special.iv(1,lDs)*sqrt(D/lam)/(d3*el)  # Time to target, starting at the target
    #return Tx + a1*a2/(2*pi)*tD*(singdif + Rdif);  # MFPT
    return Tx + a1*a2/(2*pi)*tD*(- log(el*d1/L) + RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)) #GMFPT
vec_apprx = vectorize(apprx)

## q
aflag = 1
[qg1,Xyg1] = meshgrid(linspace(0.1,0.9,50),linspace(el,L-el,50))
if aflag == 0:    
    Ta1 = vec_apprx(qg1,L-1,Xyg1,x_i,L/2)
    Ta2 = vec_apprx(qg1,Xyg1,L/2,x_i,L/2)
    np.save('fig3_data_Ta1', Ta1)
    np.save('fig3_data_Ta2', Ta2)
else:
    Ta1 = np.load('fig3_data_Ta1.npy')
    Ta2 = np.load('fig3_data_Ta2.npy')

    
##########
fig1 = figure(1,figsize=(1,1),facecolor='none',edgecolor='none')
clf()
figtext(0.02,0.9,'a',fontsize=20)
figtext(0.5,0.9,'b',fontsize=20)

ax1 = fig1.add_subplot(121)
Tmin = Ta1.min().min()
#linspace(Tmin+5,Tmin+1000,10)
Z = array([360.,400., 450., 500., 600.,700,800,900])
CS=contour(qg1,Xyg1,Ta1,Z,colors='w')
plt.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
#pcolor(qg1,Xyg1,Ta1,cmap='hot',vmin=250,vmax=800)
#figimage(Ta1,1,1,origin='lower')
imshow((Ta1),origin='lower',extent=(0.1,.9,el,L-el),aspect=0.8/(L-2*el),cmap='hot',vmin=200,vmax=1200)
axis([0.1,0.9,el,L-el])
xlabel(r'$q$',fontsize=24)
ylabel(r'$X_y$',fontsize=24)


ax2 = fig1.add_subplot(122)
Tmin = Ta2.min().min()
#linspace(Tmin+5,Tmin+1000,10)
Z = array([285.,300., 350., 400., 500., 600, 700, 800, 900])
CS=contour(qg1,Xyg1,Ta2,Z,colors='w')
plt.clabel(CS, inline=1, fontsize=10,fmt='%1.0f')
#pcolor(qg1,Xyg1,Ta2,cmap='hot',vmin=200,vmax=800)
imshow(Ta2,origin='lower',extent=(0.1,0.9,el,L-el),aspect=0.8/(L-2*el),cmap='hot',vmin=200,vmax=1000)
axis([0.1,0.9,el,L-el])
#im1.axis('equal')
xlabel(r'$q$',fontsize=24)
ylabel(r'$X_x$',fontsize=24)


savefig('fig3.pdf',format='pdf')
#savefig('fig3.eps',format='eps')
