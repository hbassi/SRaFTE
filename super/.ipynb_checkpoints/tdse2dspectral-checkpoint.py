import argparse
import numpy as np
import cupy as cp
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import cupyx.scipy.sparse as css
import cupyx.scipy.sparse.linalg as cssl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='tdse2d',
                                 description='2D TDSE spectral solver for simple scattering problem')

parser.add_argument('--outpath', required=True, help='output path')
parser.add_argument('--n', type=int, required=True, help='number of spatial grid points')
parser.add_argument('--nsteps', type=int, required=True, help='number of time steps to take')
parser.add_argument('--saveint', type=int, required=True, help='how often to save')
parser.add_argument('--saveplot', required=False, action=argparse.BooleanOptionalAction, help='whether to save plots')
parser.add_argument('--alpha', required=False, help='width parameter for Gaussian wavepacket')
parser.add_argument('--x0', required=False, help='position parameter for Gaussian wavepacket')
parser.add_argument('--p', required=False, help='momentum parameter for Gaussian wavepacket')

# actually parse command-line arguments
args = parser.parse_args()

def laplacian1D(N, dx):
    diag=np.ones(N)
    mat=sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),[-2,-1,0,1,2],N,N)
    return mat/(12*dx**2)

def laplacian2D(N, dx):
    diag=np.ones([N*N])
    mat=sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),[-2,-1,0,1,2],N,N,'dia')
    I=sp.eye(N,format='dia')
    return (sp.kron(I,mat,format='dia')+sp.kron(mat,I,format='dia'))/(12*dx**2)

# number of grid points
n = args.n

# spatial extent of grid: [[La,Lb]
La = -80
Lb = 40

# 1d grid
xgrid = cp.linspace(La, Lb, n)
xgridCPU = np.linspace(La, Lb, n)
dx = (Lb-La)/(n-1)
print("")
print("dx="+str(dx))

# 2d grid
xmat, ymat = cp.meshgrid(xgrid, xgrid)

# time step
# natural units of time
tunits = 2.4188843265857e-17
# want dt*tunits to be 2.4e-3 femtoseconds
dt = 0.01*2.4e-3*1e-15/tunits

print("dt="+str(dt))
print("dt/dx**2="+str(dt/dx**2))
print("")

# number of steps
nsteps = args.nsteps

# save 2D wave function to disk every saveint frames
saveint = args.saveint

# do you want to save contour plots?
if args.saveplot:
    saveplot = True
else:
    saveplot = False

# output path
outpath = args.outpath

# set up one electron in the soft Coulomb external potential
ham1d = -0.5*laplacian1D(n, dx) - sp.spdiags( np.array( [((xgrid.get() + 10)**2 + 1)**(-0.5)] ), 0, n, n )

# compute ground state
evals, evecs = spl.eigsh(ham1d, k=1, which='SA')

# normalization constant is 1/sqrt(dx)
# numgs = numerical ground state
numgs = dx**(-0.5)*evecs[:,0]
print("ground state normalization = " + str(np.sum(np.square(np.abs(numgs)))*dx))

# check that it's an eigenfunction
print("eigenfunction test; the following should be close to 0:")
print(np.linalg.norm( ham1d@numgs - evals[0]*numgs ))

# define Gaussian wavepacket
def phiWP(x,alpha,x0,p):
    phi = ((2*alpha/np.pi)**(0.25))*np.exp(-alpha*(x-x0)**2 + (1j)*p*(x-x0)) 
    return phi

# initial condition
# psiold = cp.zeros(n**2, dtype=cp.complex128)
if args.alpha:
    myalpha = float(args.alpha)
else:
    myalpha = 0.1

if args.x0:
    myx0 = float(args.x0)
else:
    myx0 = 10.0

if args.p:
    myp = float(args.p)
else:
    myp = -1.8

print("")
print("Gaussian wavepacket parameters")
print("alpha = " + str(myalpha))
print("x0 = " + str(myx0))
print("p = " + str(myp))
print("")

phiWPvec = phiWP(xgrid.get(),myalpha,myx0,myp)
psiold = cp.array( ( 1.0/np.sqrt(2.0)*(np.outer(phiWPvec, numgs) + np.outer(numgs, phiWPvec)) ).reshape((-1)) )

# make sure things are normalized
normalizer = ( np.sum(np.square(np.abs(psiold))*dx*dx ) )**(-1/2)
psiold *= normalizer
print("normalization test; the following should be close to 1:")
print(np.sum(np.square(np.abs(psiold))*dx*dx))

# set up array needed for iteration
psinew = cp.zeros(n**2, dtype=cp.complex128)

# form Laplacian operator IN FOURIER SPACE
L = Lb - La
kvec = (2*np.pi/L)*np.arange(-n/2, n/2)
kvec = np.fft.fftshift(kvec)
k1, k2 = np.meshgrid(kvec,kvec,indexing='ij')
lap = -(k1**2 + k2**2)

lf = 1j*dt/2.0
# Fourier symbol of kinetic *half* dt step propagator 
jkinhalf = cp.exp((lf/2.0)*cp.asarray(lap))

# evolve forward in time by half step of kinetic
def evolvekinhalf(jpsi):
    # compute 2D FFT
    jpsihat = cp.fft.fft2(jpsi.reshape((n,n)))
    # compute 2D IFFT
    return cp.fft.ifft2(jpsihat*jkinhalf)

# potential energy propagator
vmat = -((xmat + 10)**2 + 1)**(-0.5) - ((ymat + 10)**2 + 1)**(-0.5) + ((xmat-ymat)**2 + 1)**(-0.5)
gpupotprop = cp.exp((-1j)*dt*vmat)

print("")
print("Beginning propagation!")
print("")
print("[time step, normalization constant]")
for j in range(nsteps):
    
    # operator splitting time step
    psinew[:] = evolvekinhalf(gpupotprop * evolvekinhalf(psiold)).reshape((-1))
    
    # save the 2D wave function itself
    psinewM = psinew.get().reshape((n,n))

    # save wave function
    if j % saveint == 0:
        np.save(outpath + 'psi' + str(j).zfill(5) + '.npy', psinewM)
        # monitor normalization, should be 1.0 or super close
        normconst = cp.sum(cp.square(cp.abs(psinew))*dx*dx).item()
        print( [j, normconst] )
        if saveplot:
            plt.figure(figsize=(5,5))
            plt.contourf(np.abs(psinewM), levels=40)
            plt.title(r'$|\Psi(x, t)|$ at t = ' + "{:.4e}".format(j*dt))
            plt.savefig(outpath + 'psicontour' + str(j).zfill(5) + '.png')
            plt.close()

    psiold[:] = psinew

print("")
