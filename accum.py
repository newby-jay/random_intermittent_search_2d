from pylab import *
from IPython import genutils
import scipy.weave
L = 5.0;
Bx = 0.5;
By = 0.1;
y0 = 0;
y = linspace(-L,L,100);
x0 = linspace(-L,L,20);
My = y.shape[0];
Nx = x0.shape[0];
Nij = 150;
GAMMAX = zeros((My,Nx));
PI = pi;

## for n in arange(Nx):
##     gammax = 0;    
##     for i in arange(Nij):
##         for j in arange(1,Nij):            
##             phix0 = cos(pi/(4.0*L)*(2*i+1)*(x0[n]+L))*sin(pi*j/(2.0*L)*(y0-L));
##             lambdaij = pi**2/L**2 *((2*i+1)**2/16.0*Bx + j**2/4.0*By);
##             gammax = gammax + (2*i+1)/lambdaij*phix0*(-1.0)**i*sin(pi*j/(2*L)*(y-L));
##     GAMMAX[:,n] = pi*Bx/(4.0*L)*gammax;



t0=genutils.clock()
code = """
double phix0, lambdaij, const_1;
for (int ix=1;ix<Nx-1;ix++){
   for (int i=0;i<Nij;i++){
      for (int j=1;j<Nij;j++){
         phix0 = cos(PI/(4.0*L)*(2*i+1)*(x0[ix]+L))*sin(PI*j/(2.0*L)*(y0-L));
         lambdaij = (PI*PI)/(L*L) *(pow(2.0*i+1,2.0)/16.0*Bx + j*j/4.0*By);
         const_1 = (2*i+1)/lambdaij*phix0*pow(-1.0,i);
         for (int k=0;k<My-1;k++){            
            GAMMAX[k,ix] = GAMMAX[k,ix] + const_1*sin(PI*j/(2*L)*(y[k]-L));
          }
      }
   }
}

"""
scipy.weave.inline(code,['L','Bx','By','y0','My','Nx','Nij','x0','y','GAMMAX','PI'])
t1 = genutils.clock()


print t1-t0
## figure(1)
plot(GAMMAX);

