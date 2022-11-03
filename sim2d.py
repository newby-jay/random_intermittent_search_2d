from pylab import *
from IPython import genutils
import scipy.weave
from scipy import random
from scipy import special
## 2D mc simulation of intermittent search
## p(h+,h-,v+,v-,0)



## Parameters
L = 5.0; # domain is (0,L)x(0,L)
# initial condition
ti = 0.;
yi = L/2.;
xi = 0.1;
si = 0;
thetai = 0;
# target
el = .2;
Xx = L-1;
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


code5s = """    
    double x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd;
    int n, s, s0, j;
    double alphac = alpha;
    double betac = beta;
    const gsl_rng_type * Trng;
    int sd = seed;
    gsl_rng * r;
    gsl_rng_env_setup();
    Trng = gsl_rng_default;
    r = gsl_rng_alloc(Trng);
    gsl_rng_set(r,sd);
        
    double nt = 0;
    double t_tot = 0;
    double t_tot_square = 0;
    while (nt<NP) {
        nt += 1;
        x = xi; y = yi; t = ti; s = si;
        int exit = 1;
            while (exit == 1){
                x0=x; y0=y; t0=t; s0=s;                
                
                R1 = gsl_rng_uniform_pos(r);
                // The searching state
                if (s0==4){ 
                    R2 = gsl_rng_uniform_pos(r);
                    for (j = 0; j <= 4 ; j++){ 
                        if ( R1 < Qd[j] ){ 
                            s = j;
                            j = 5;
                        }
                    }
                    // next time
                    tnext = -log(R2)/alphac;
                    t = t0 + tnext;
                    // target
                    d = sqrt((x0-Xx)*(x0-Xx) + (y0-Xy)*(y0-Xy));
                    if ( d<el ) {
                        R3 = gsl_rng_uniform_pos(r);
                        t_absorb = -log(R3)/k;
                        if (t_absorb<tnext){
                            exit = 0;                          
                            t = t0 + t_absorb;
                        }
                    }                                                            
                }
                // Moving states
                else {
                   s = 4;
                   tnext = -log(R1)/betac;
                   t = t0 + tnext;
                   x = x0 + vx[s0]*tnext;
                   y = y0 + vy[s0]*tnext;
                }

               //reflecting boundary conditions 
                if (x<0){
                    x = 0;
                    s = 0;
                    t = t0 + x0;
                }
                else if (x>L){
                    x = L;
                    s = 1;
                    t = t0 + (L-x0);
                }
                
                if (y<0){
                    y = 0;
                    s = 2;
                    t = t0 + y0;
                }
                else if (y>L){
                    y = L;
                    s = 3;
                    t = t0 + (L-y0);
                }                                
        }
        t_tot +=  t;
        t_tot_square += t*t;
    }
    T = t_tot/NP;
    Tstd = sqrt(t_tot_square/NP - T*T);
    ans[0]=T; ans[1]=Tstd;
    gsl_rng_free(r);
"""
def mc_sim_5s(ti,xi,yi,si,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NP):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(code5s,
                   ['ti','xi','yi','si','L','el','Xx','Xy','Qd','vx','vy','alpha','beta','k','NP','seed','ans'],
                   headers = ['<math.h>','<gsl/gsl_rng.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans

coderd = """    
    double theta, theta0, x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd;
    int n, s0, s, j;
    double alphac = alpha;
    double betac = beta;
    const gsl_rng_type * Trng;
    int sd = seed;
    gsl_rng * r;
    gsl_rng_env_setup();
    Trng = gsl_rng_default;
    r = gsl_rng_alloc(Trng);
    gsl_rng_set(r,sd);
        
    double nt = 0;
    double t_tot = 0;
    double t_tot_square = 0;
    while (nt<NP) {
        nt += 1;
        x = xi; y = yi; t = ti; theta = thetai; s = si;
        int exit = 1;
            while (exit == 1){
                x0=x; y0=y; t0=t; theta0=theta; s0=s;                
                
                R1 = gsl_rng_uniform_pos(r);
                // The searching state
                if (s0 == 1){
                    s = 0;
                    R2 = gsl_rng_uniform_pos(r);
                    theta = 2*M_PI*R2;
                    // next time
                    tnext = -log(R2)/alphac;
                    t = t0 + tnext;
                    // target
                    d = sqrt((x0-Xx)*(x0-Xx) + (y0-Xy)*(y0-Xy));
                    if ( d<el ) {
                        R3 = gsl_rng_uniform_pos(r);
                        t_absorb = -log(R3)/k;
                        if (t_absorb<tnext){
                            exit = 0;                          
                            t = t0 + t_absorb;
                        }
                    }                                                            
                }
                // Moving states
                else {
                   s = 1;
                   tnext = -log(R1)/betac;
                   t = t0 + tnext;
                   x = x0 + v*cos(theta0)*tnext;
                   y = y0 + v*sin(theta0)*tnext;
                }

               //reflecting boundary conditions               
               
                if (x < 0){
                    s = 0;
                    x = 0;
                    theta = fmod(theta0+M_PI,2*M_PI);
                    y = y0 - x0*tan(theta0);//(0-x0)/(v*cos(theta0))*v*sin(theta0);
                    t = t0 + (0-x0)/(v*cos(theta0));
                }  
                else if (x > L){
                    s = 0;
                    x = L;
                    theta = fmod(theta0+M_PI,2*M_PI);
                    y = y0 + (L-x0)*tan(theta0);//(L-x0)/(v*cos(theta0))*v*sin(theta0);
                    t = t0 + (L-x0)/(v*cos(theta0));
                }
                
                if (y < 0){
                    s = 0;
                    y = 0;
                    theta = fmod(theta0+M_PI,2*M_PI);
                    x = x0 -y0/tan(theta0);//((0-y0)/(v*sin(theta0))) * v*cos(theta0);
                    t = t0 + (0-y0)/(v*sin(theta0));
                }
                else if (y > L){
                    s = 0;
                    y = L;
                    theta = fmod(theta0+M_PI,2*M_PI);
                    x = x0 + (L-y0)/tan(theta0);//(L-y0)/(v*sin(theta0))*v*cos(theta0);
                    t = t0 + (L-y0)/(v*sin(theta0));
                }                                
        }
        t_tot +=  t;
        t_tot_square += t*t;
    }
    T = t_tot/NP;
    Tstd = sqrt(t_tot_square/NP - T*T);
    ans[0]=T; ans[1]=Tstd;
    gsl_rng_free(r);
"""

def mc_sim_rd(ti,xi,yi,thetai,si,L,el,Xx,Xy,v,alpha,beta,k,NP):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(coderd,
                   ['ti','xi','yi','thetai','si','L','el','Xx','Xy','v','alpha','beta','k','NP','seed','ans'],
                   headers = ['<math.h>','<gsl/gsl_rng.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans

## MC sim
Ni = 20;
Nj = 2;
T = zeros((Ni,Nj));
Var = zeros((Ni,Nj));
alphav = linspace(1.0,5.0,Ni);
betav = linspace(3.0,6.0,Nj);
        
mcflag = 1
NP = 5000
if mcflag==0:
    for i in arange(Ni):
        alpha = alphav[i]
        for j in arange(Nj):
            beta = betav[j]                    
            sim = zeros((1,2))        
            #sim = mc_sim_5s(ti,xi,yi,si,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NP)
            sim = mc_sim_rd(ti,xi,yi,thetai,si,L,el,Xx,Xy,1,alpha,beta,k,NP)
            T[i,j] = sim[0,0]
            Var[i,j] = sim[0,1]
    np.save('mc_data', T)
else:
    T = np.load('mc_data.npy')

## Analytical
def RegP(xi1,eta1,xi2,eta2,a1,a2):
    H0 = 1/12.*(4. - 6.*(abs(eta1-eta2)+eta1+eta2)/a2 + 3.*((eta1-eta2)**2 + (eta1+eta2)**2)/a2**2);
    z1 = exp(1j*pi*(xi1+xi2)/a2);
    z2 = exp(1j*pi*(xi1-xi2)/a2);
    zeta1 = exp(-pi/a2*abs(eta1+eta2));
    zeta2 = exp(-pi/a2*abs(eta1-eta2));
    sigma1 = exp(-pi/a2*(2*a1-abs(eta1+eta2)));
    sigma2 = exp(-pi/a2*(2*a1-abs(eta1+eta2)));
    if xi1==xi2 :
        arg1 =  1   
    else:
        arg1 = abs(1-z2*zeta2)/sqrt((xi1-xi2)**2+(eta1-eta2)**2)
    arg2 = abs(1-z2*sigma2)*abs(1-z2*sigma1)*abs(1-z1*sigma2)*abs(1-z1*sigma1)*abs(1-z2*zeta1)*abs(1-z1*zeta2)*abs(1-z1*zeta1);
    R = 2.*pi*H0 - log(arg1) - log(arg2);
    return R;
def apprx(alpha,beta):
    a1 = 1./sqrt(2*(1-q));
    a2 = 1./sqrt(2*q);
    xi0 = a2*xi/L;  # xi is the initial value of x, but xi0 is the rescaled (greek xi) initial value of x
    eta0 = a1*yi/L;
    Xxi =  a2*Xx/L; # Xxi referes to the rescaled (greek xi) target coordinate
    Xeta = a1*Xy/L;
    a = alpha/(alpha+beta);
    b = beta/(alpha+beta);
    lam = k*b;  #  effective absorbtion rate
    D = a/(2.*beta);  # effective diffusivity
    tdist = sqrt((xi0-Xxi)**2 + (eta0-Xeta)**2);  # distance to target
    d = 1/2.*(a1+a2);  # logarithmic capacitance of the target

    Rdif = RegP(Xxi,Xeta,Xxi,Xeta,a1,a2)  -  RegP(xi0,eta0,Xxi,Xeta,a1,a2); # Regular part of the Green's function
    singdif = log(tdist) - log(el*d/L);  ## singular part of the Green's function
    lDs = el*d/sqrt(D/lam);  # target radius / absorbtion length scale
    tD = L**2/D       # diffusion time scale to explore the domain
    Tx = 1./lam + a1*a2/(2*pi)*tD *  special.iv(0,lDs)/special.iv(1,lDs)*sqrt(D/lam)/el;  # Time to target, starting at the target
    return Tx + a1*a2/(2*pi)*tD*(singdif + Rdif);  # MFPT
vec_apprx = vectorize(apprx)
[beta_a,alpha_a] = meshgrid(linspace(betav[0],betav[-1],Nj),linspace(alphav[0],alphav[-1]));
Ta = vec_apprx(alpha_a,beta_a);


#imshow(T,interpolation='bilinear',origin='lower',extent=(-3,3,-3,3));
#figure(1);
#imshow(log(Ta),cmap=cm.gray,interpolation='bilinear',origin='lower',extent=(betav[0],betav[-1],alphav[0],alphav[-1]));


## fig2=figure(2)
## fig2.add_subplot(121)
## plot(alpha_a,Ta)
## plot(alphav,T,'o')
## xlabel(r'$\alpha$',fontsize=24)
## ylabel(r'$T$',fontsize=24)
## fig2.add_subplot(122)
## plot(betav,transpose(Ta))
## plot(betav,transpose(T),'o')


fig3 = figure(3)


