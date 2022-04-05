import math
import numpy as np
from scipy.special import erf
from scipy.special import hyp1f1

impAtoms=['H','He','Li','Be','B','C','N','O','F','Ne'] #implemented atoms
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
    return numAtoms,atoms,np.array(cords,dtype=np.float)





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
    redexp=gA[0]*gB[0]/p
    r=redexp*(3-2*redexp*mag)/p**1.5*K
    #print(redexp)
    return r

def potential(gA,gB,Rc,Zc):#Nuclear Electron what index of atom list 
    """Just defined for electrons in 1/2s for now"""
    mag,K,p,Rp=gaussProd(gA,gB)
    #Rc=cords[atomnum]
    #Zc=charges[atoms[atomnum]] -> send as param
    return -2*K/math.sqrt(math.pi)*Zc/p*F0(p*np.linalg.norm(Rp-Rc)**2)

def multiOverlap(gA,gB,gC,gD,bN=0):
    magAB,KAB,pAB,RpAB=gaussProd(gA,gB)
    magCD,KCD,pCD,RpCD=gaussProd(gC,gD)
    r=2/math.sqrt(math.pi)*(pAB*pCD*(pAB+pCD)**0.5)**-1*KAB*KCD
    if bN==0:
        r*=F0(pAB*pCD/(pAB+pCD)*np.linalg.norm(RpAB-RpCD)**2)
    else:
        r*=boysN(pAB*pCD/(pAB+pCD)*np.linalg.norm(RpAB-RpCD)**2,bN)
    return r


def difGuess(pold,p,numBasis):
    return sum([numBasis**-2*(pold[i,j]-p[i,j])**2 for i,j in zip(range(numBasis),range(numBasis))])**0.5



def sto3G(atoms,cords,N,eps=1e-4):
    """Calculate steady state Hartree Fock energy calculations for N electrons and 3Gaussian basis set"""
    assert len(atoms)==len(cords)
    
    numBasis=sum((maxQNum[a] for a in atoms)) # don't use quantum num n
    
    ol=np.zeros((numBasis,numBasis))
    kin=np.zeros((numBasis,numBasis))
    pot=np.zeros((numBasis,numBasis))
    MET=np.zeros((numBasis,numBasis,numBasis,numBasis))

    for ia,atoma in enumerate(atoms):
        Za=charges[atoma] #charge 
        Ra=cords[ia] #center
        for oa in range(maxQNum[atoma]):
            coefa=gcoef[oa]
            za=zeta[atoma][oa] #zeta
            ga=alpha[oa]*za**2#gaussfactor
            for a in range(3): ###in future design data structure for basis sets and make this function STO(n)G
                for ib,atomb in enumerate(atoms):
                    Zb=charges[atomb]
                    Rb=cords[ib]
                    for ob in range(maxQNum[atomb]):
                        coefb=gcoef[ob]
                        zb=zeta[atomb][ob] #zeta
                        gb=alpha[ob]*zb**2#gaussfactor
                        for b in range(3):
                            i0=(ia+1)*(oa+1)-1
                            i1=(ib+1)*(ob+1)-1
                            ol[i0,i1]+=coefa[a]*coefb[b]*overlap((ga[a],Ra),(gb[b],Rb))
                            kin[i0,i1]+=coefa[a]*coefb[b]*kinetic((ga[a],Ra),(gb[b],Rb))
                            for i in range(len(atoms)):
                                pot[i0,i1]+=coefa[a]*coefb[b]*potential((ga[a],Ra),(gb[b],Rb),cords[i],charges[atoms[i]])
                            for ic,atomc in enumerate(atoms):
                                Zc=charges[atomc]
                                Rc=cords[ic]
                                for oc in range(maxOrbitals[atomc]):
                                    coefc=gcoef[oc]
                                    zc=zeta[atomc][oc] #zeta
                                    gc=alpha[oc]*zc**2#gaussfactor
                                    for c in range(3):
                                        for id,atomd in enumerate(atoms):
                                            Zd=charges[atomd]
                                            Rd=cords[id]
                                            for od in range(maxOrbitals[atomd]):
                                                coefd=gcoef[od]
                                                zd=zeta[atomd][od]
                                                gd=alpha[od]*zd**2
                                                for d in range(3):
                                                    i2=(ic+1)*(oc+1)-1
                                                    i3=(id+1)*(od+1)-1
                                                    MET[i0,i1,i2,i3]+=coefa[a]*coefb[b]*coefc[c]*coefd[d]*multiOverlap((ga[a],Ra),(gb[b],Rb),(gc[c],Rc),(gd[d],Rd)) # this line needs changed -> what determines the boys factor? any existence of p orb?
    Hcore=kin+pot#core hamiltonian, doesn't change in iteration
    ovals,ovecs=np.linalg.eig(ol)
    odiag=np.dot(ovecs.T,np.dot(ol,ovecs))
    orootdiag=np.diag(np.diagonal(odiag)**-0.5)
    X=np.dot(ovecs,np.dot(orootdiag,ovecs.T)) #symmetrically orthogonalized

    P=np.zeros((numBasis,numBasis))
    pold=np.zeros((numBasis,numBasis))
    ps=[]

    thresh=1
    while thresh>eps:
        G=np.zeros((numBasis,numBasis))
        for i in range(numBasis):
            for j in range(numBasis):
                for x in range(numBasis):
                    for y in range(numBasis):
                        G[i,j]+=P[x,y]*MET[i,j,y,x]-.5*MET[i,x,y,j]#pg 175
        fock=Hcore+G
        fp=np.dot(X.T,np.dot(fock,X))
        evalfp,eigfp=np.linalg.eig(fp)
        asc=evalfp.argsort()
        evalfp=evalfp[asc]
        eigfp=eigfp[:,asc]
        C=np.dot(X,eigfp)
      # print(f"C is {C}")
        for i in range(numBasis):
             for j in range(numBasis):
                for k in range(N//2):
                    P[i,j]=C[i,k]*C[j,k]# final wave function / orbital density mat
        ps.append(P)
        thresh=difGuess(pold,P,numBasis)
        pold=P.copy()

    #nuclear repulsion term
    repulse=0
    for ia,atoma in enumerate(atoms):
        for ib,atomb in enumerate(atoms):
            if ia!=ib:
                zA=charges[atoma]
                zB=charges[atomb]
                Ra=cords[ia]
                Rb=cords[ib]
                repulse+=zA*zB/np.linalg.norm(Ra-Rb)
    repulse*=.5
    #print(X)
    #print(f"\n{MET}")
    return ps,evalfp,repulse,Hcore # list of density matrices + eigenvalues = E levels + nuclear repulsion term
        


    