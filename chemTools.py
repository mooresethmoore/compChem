import math
import numpy as np
from scipy import erf
from scipy.special import hyp1f1

impAtoms=['H','He','Li','Be','B','C','N','O'.'F','Ne'] #implemented atoms
charges={impAtoms[i]:i+1 for i in range(len(impAtoms))}
maxQNum = {'H':1,'He':1,'Li':2,'Be':2,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2}#Maximum quantum number -> web scrape from periodic table dat and fill up through 3p
maxOrbitals={'H':1,'He':1,'Li':2,'Be':2,'B':3,'C':3,'N':3,'O':3,'F':3,'Ne':3}
#implementations of calculating proper gauss coefs for STO(n)G

gcoef=np.array([[.444635,.535328,.154329],
[.700115,.399513,-.0999672],
[.391957,.607684,.155916]])#1s,2s,2p linear combination of gauss coefs for 3G -> pg 185

#shared 2sp alpha + zetas
alpha=np.array([[.109818,.405771,2.22766],
[.0751386,.231031,.994203],
[.0751386,.231031,.994203]])
zeta={'H':[1.24],'He':[2.0925],'Li':[2.69,.75],'Be':[3.68,1.1],'B':[4.68,1.45,1.45],'C':[5.67,1.72,1.72],'N':[6.67,1.95,1.95],'O':[7.66,2.25,2.25],'F':[8.65,2.55,2.55]}



def parseXYZ(file):
    numAtoms=0
    atoms=[]
    cords=[]
    with open(file) as f:
        numAtoms=int(f.readline())
        for line in f:
            sp=line.split()
            if len(sp)>0: # Allow future implementations for avoiding the coordinates of the first atom for origin
                atoms.append(sp[0])
                cords.append(sp[1:])
    return numAtoms,atoms,cords





def F0(t):#boys 0 function for 1s,2s orbitals
    r=1
    if t != 0:
        r=.5*math.sqrt(math.pi/t)*erf(math.sqrt(t))
    return r

def boysN(t,n):
    """ Attempted calc of Boys function for higher orbitals: uses relationship to
    Confluent hypergeometric function """

    return hyp1f1(n+.5,n+1.5,-t)/(2*n+1)

def gaussProd(gA,gB):
    """ each gaussian object is (exponent,center) tuple """
    a,Ra=gA
    b,Rb=gB
    mag=np.linalg.norm(Ra-Rb)**2 #Length from Ra to Rb
    p=a+b
    K=(4*a*b)**.75*np.exp(-a*b/p*mag) # Proportionality constant with common normalization term -> missing 1/(pi)**1.5 term
    Rp=(a*Ra+b*Rb)/p
    return mag,K,p,Rp

def overlap(gA,gB):
    """2 center overlap integral  
    page 412"""
    mag,K,p,Rp=gaussProd(gA,gB)
    return K/p**1.5 #pis cancel

def kinetic(gA,gB):
    mag,K,p,Rp=gaussProd(gA,gB)
    redexp=gA[0]*gA[1]/p
    return redexp*(3-2*redexp*mag)/p**1.5*K

def nucAttr(gA,gB,Rc,Zc):# what index of atom list
    """Just defined for electrons in 1/2s for now"""
    mag,K,p,Rp=gaussProd(gA,gB)
    #Rc=cords[atomnum]
    #Zc=charges[atoms[atomnum]] -> send as param
    return -2*K/math.sqrt(math.pi)*Zc/p*F0(p*np.linalg.norm(Rp-Rc)**2)

def multiOverlap(gA,gB,gC,gD):
    magAB,KAB,pAB,RpAB=gaussProd(gA,gB)
    magCD,KCD,pCD,RpCD=gaussProd(gC,gD)





def sto3G(atoms,cords,N):
    """Calculate steady state Hartree Fock energy calculations for N electrons and 3Gaussian basis set"""
    assert len(atoms)==len(cords)
    
    numBasis=sum((maxQNum(a) for a in atoms))


    