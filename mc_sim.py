from pylab import *
from IPython import genutils
import scipy.weave
from scipy.weave import converters
from scipy import random
from scipy import special


coderd = """    
    double theta, theta0, x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd;
    int n, s0, s, j;
    double alphac = alpha;
    double betac = beta;
    int NP = NI;
    double ti = t_i;
    double xi = x_i;
    double yi = y_i;
    double thetai = theta_i;
    int si = s_i;

    
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
                    tnext = -log(R1)/alphac;
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

def mc_sim_rd(t_i,x_i,y_i,theta_i,s_i,L,el,Xx,Xy,v,alpha,beta,k,NI):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(coderd,
                   ['t_i','x_i','y_i','theta_i','s_i','L','el','Xx','Xy','v','alpha','beta','k','NI','seed','ans'],
                   headers = ['<math.h>','<gsl/gsl_rng.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans
#mc_sim_rd_vec = vectorize(mc_sim_rd);

coderd_avg = """    
    double theta, theta0, x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd;
    int n, s0, s, j;
    double alphac = alpha;
    double betac = beta;
    int NP = NI;
    double ti = t_i;
    double xi, yi, thetai;
    int si = s_i;

    
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
        R1 = gsl_rng_uniform_pos(r);
        thetai = 2*M_PI*R1;
        R1 = gsl_rng_uniform_pos(r);
        xi = L*R1;
        R1 = gsl_rng_uniform_pos(r);
        yi = L*R1;        
        R1 = gsl_rng_uniform_pos(r);
        if (R1<betac/(alphac+betac)){si = 1;} else { si = 0; }
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
                    tnext = -log(R1)/alphac;
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

def mc_sim_rd_avg(t_i,s_i,L,el,Xx,Xy,v,alpha,beta,k,NI):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(coderd_avg,
                   ['t_i','s_i','L','el','Xx','Xy','v','alpha','beta','k','NI','seed','ans'],
                   headers = ['<math.h>','<gsl/gsl_rng.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans
#mc_sim_rd_avg_vec = vectorize(mc_sim_rd);

code5s = """    
    double x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd;
    int n, s, s0, j;
    
    double alphac = alpha;
    double betac = beta;
    int NP = NI;
    double ti = t_i;
    double xi = x_i;
    double yi = y_i;
    int si = s_i;
    double Lc = L;
    double elc = el;
    double Xxc = Xx;
    double Xyc = Xy;
    double kc = k;
    
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
                    d = sqrt((x0-Xxc)*(x0-Xxc) + (y0-Xyc)*(y0-Xyc));
                    if ( d<elc ) {
                        R3 = gsl_rng_uniform_pos(r);
                        t_absorb = -log(R3)/kc;
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
                else if (x>Lc){
                    x = Lc;
                    s = 1;
                    t = t0 + (Lc-x0);
                }
                
                if (y<0){
                    y = 0;
                    s = 2;
                    t = t0 + y0;
                }
                else if (y>Lc){
                    y = Lc;
                    s = 3;
                    t = t0 + (Lc-y0);
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
def mc_sim_5s(t_i,x_i,y_i,s_i,L,el,Xx,Xy,Qd,vx,vy,alpha,beta,k,NI):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(code5s,
                   ['t_i','x_i','y_i','s_i','L','el','Xx','Xy','Qd','vx','vy','alpha','beta','k','NI','seed','ans'],
                   headers = ['<math.h>','<gsl/gsl_rng.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans
#mc_sim_5s_vec = vectorize(mc_sim_5s)


code5s_bias = """    
    double x, x0, y, d, y0, t, t0, R1, R2, R3, tnext, t_absorb, T, Tstd, va;
    int n, s, s0, j;
    
    double alphac = alpha;
    double betac = beta;
    int NP = NI;
    double ti = t_i;
    double xi = x_i;
    double yi = y_i;
    int si = s_i;
    double Lc = L;
    double vdriftx = vdx;
    double vdrifty = vdy;
    
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
            while ( exit == 1){
                x0=x; y0=y; t0=t; s0=s;                
                
                R1 = gsl_rng_uniform_pos(r);
                // The searching state
                if (s0==4){ 
                    R2 = gsl_rng_uniform_pos(r);
                    for (j = 0; j <= 4 ; j++){ 
                        if ( R2 < Qd[j] ){ 
                            s = j;
                            j = 5;
                        }
                    }
                    // next time
                    tnext = -log(R1)/alphac;
                    t = t0 + tnext;                                        
                }
                // Moving states
                else {
                   s = 4;
                   tnext = -log(R1)/betac;
                   t = t0 + tnext;
                   x = x0 + vx[s0]*tnext;
                   y = y0 + vy[s0]*tnext;
                }

               //moving boundary condition
               d = sqrt( (x-vdriftx*t)*(x-vdriftx*t) + y*y );
               //cout << "x = " << x;
               //cout << " y = \\n " << y
               if ( d>Lc ){                  
                  exit = 0;
                  t = t0 + tnext/2;
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
def mc_sim_5s_bias(t_i,x_i,y_i,s_i,L,Qd,vx,vy,vdx,vdy,alpha,beta,NI):    
    randpow = ceil(random.rand()*16)
    seed = floor(random.rand()*10**randpow)
    ans = zeros((1,2))
    timer0 = genutils.clock()
    scipy.weave.inline(code5s_bias,
                   ['t_i','x_i','y_i','s_i','L','Qd','vx','vy','vdx','vdy','alpha','beta','NI','seed','ans'],                   
                   headers = ['<math.h>','<gsl/gsl_rng.h>','<iostream.h>'],
                   libraries=['gsl'],
                   include_dirs=['/usr/local/include'],
                   library_dirs=['/usr/local/lib'])
    timer1 = genutils.clock()
    print timer1-timer0
    return ans



if __name__ == '__main__':   
    L = 1.0 
    t_i = 0.
    y_i = 0.
    x_i = 0.
    s_i = 0
    theta_i = 0
    # target
    el = .25
    Xx = L/2
    Xy = L/2.
    # rates
    k = 1
    beta = 1.
    alpha = 1.
    q = 0.5
    Qd = array([q/2.,q,q+(1.-q)/2.,1.])
    vx = array([1.,-1.,0.,0.])
    vy = array([0.,0.,1.,-1.])
    NI = 1
    sol = mc_sim_5s_bias(t_i,x_i,y_i,s_i,L,Qd,vx,vy,0,0,alpha,beta,NI)
