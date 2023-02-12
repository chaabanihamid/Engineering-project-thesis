#!/usr/bin/python
"""
Updated on 31 October 2020 

@author: Noureddine SEMANE, Correction Modelling Project EHTP 2020
"""


#Import Python moduls
import numpy as np
import math
import sys
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
#np.set_printoptions(precision=4)
q = input("###############################################################\n"
          "Correction Modelling Project\n"
          "October 2020 - December 2020\n"
          "@author: Pr. Noureddine SEMANE EHTP\n"
          "###############################################################\n\n"
          "Project number ?| 0 for the list of the projects ")

while "0" == q:
    print("Project list :\n1\n2\n")
    q = input("Give the project number ?|  0 for the list of the projects ")
if "1" == q:
    eb = np.zeros((1,1))
    eb[0]=0.4
    print('wind background error\n',np.round(eb,4))
    R= np.zeros((1,1))
    R[0][0]=0.01
    print('obs error variance\n',np.round(R,4))
    xb = np.zeros((4,1))
    xb[0]=1
    xb[1]=2
    xb[2]=3
    xb[3]=1.4
    xt = np.zeros((4,1))
    xt[0]=1
    xt[1]=2
    xt[2]=3
    xt[3]=1
    B= np.zeros((4,4))
    B[0][0]=R[0][0]
    B[1][1]=R[0][0]
    B[2][2]=R[0][0]
    B[3][3]=1
    print('Background error covariance at time t0\n',np.round(B,4))
    HT = np.zeros((4,1))
    HT[0]=1
    H=np.transpose(HT)
    print('Obs. operator \n',np.round(H,4))
    M= np.zeros((4,4))
    M[0][0]=1
    M[0][1]=-0.5*xb[3]
    M[0][2]=0.5*xb[3]
    M[0][3]=0
    M[1][0]=0.5*xb[3]
    M[1][1]=1
    M[1][2]=-0.5*xb[3]
    M[1][3]=0
    M[2][0]=-0.5*xb[3]
    M[2][1]=0.5*xb[3]
    M[2][2]=1
    M[2][3]=0
    M[3][0]=0
    M[3][1]=0
    M[3][2]=0
    M[3][3]=1
    print('Model\n',np.round(M,4))
    Mt= np.zeros((4,4))
    Mt[0][0]=1
    Mt[0][1]=-0.5*xt[3]
    Mt[0][2]=0.5*xt[3]
    Mt[0][3]=0
    Mt[1][0]=0.5*xt[3]
    Mt[1][1]=1
    Mt[1][2]=-0.5*xt[3]
    Mt[1][3]=0
    Mt[2][0]=-0.5*xt[3]
    Mt[2][1]=0.5*xt[3]
    Mt[2][2]=1
    Mt[2][3]=0
    Mt[3][0]=0
    Mt[3][1]=0
    Mt[3][2]=0
    Mt[3][3]=1
    print('True model\n',np.round(Mt,4))
    L= np.zeros((4,4))
    L[0][0]=1
    L[0][1]=-0.5*xb[3]
    L[0][2]=0.5*xb[3]
    L[0][3]=-0.5*(xb[1]-xb[2])
    L[1][0]=0.5*xb[3]
    L[1][1]=1
    L[1][2]=-0.5*xb[3]
    L[1][3]=-0.5*(xb[2]-xb[0])
    L[2][0]=-0.5*xb[3]
    L[2][1]=0.5*xb[3]
    L[2][2]=1
    L[2][3]=-0.5*(xb[0]-xb[1])
    L[3][0]=0
    L[3][1]=0
    L[3][2]=0
    L[3][3]=1
    print('Tangent linear model\n',np.round(L,4))
    print('Adjoint model\n',np.round(np.transpose(L),4))
    yo= np.zeros((1,1))
    yo[0]=np.dot(np.dot(H,Mt),xt)
    print('Observation\n',np.round(yo,4))
    d = np.zeros((1,1))
    d[0]=yo[0]-np.dot(np.dot(H,M),xb)
    #The innovation can also be given by d[0]=0.5*eb[0]*(xb[1]-xb[2])
    print('Innovation\n',np.round(d,4))
    H=np.dot(H,L)
    HT=np.transpose(H)
    BHT=np.dot(B,HT)
    HB=np.dot(H,B)
    HBHT=np.dot(HB,HT)
    K=np.dot(BHT,inv(np.matrix(HBHT+R)))
    xa= np.zeros((4,1))
    xa=xb+np.dot(K,d)
    print('Background at time t0\n',np.round(xb,4))
    print('4D-Var Analysis at time t0\n',np.round(xa,4))
    print('Truth at time t0\n',np.round(xt,4))
    xb1= np.zeros((4,1))
    xb1=np.dot(M,xb)
    xa1= np.zeros((4,1))
    xa1=xb1+np.dot(L,np.dot(K,d))
    print('Background at time t1\n',np.round(xb1,4))
    print('4D-Var Analysis at time t1\n',np.round(xa1,4))
    xt1= np.zeros((4,1))
    xt1=np.dot(Mt,xt)
    print('Truth at time t1\n',np.round(xt1,4))
# plot the data
    fig1 = plt.figure(figsize = (8,8))
    plt.subplots_adjust(hspace=0.4)
    x = np.array([1,2,3])
    z = np.array([0.5,1,2,3,3.5])
    p1 = plt.subplot(2,1,1)
    a= np.zeros((3,1))
    a[0] =xb[0]  
    a[1] =xb[1]  
    a[2] =xb[2]  
    aa= np.zeros((5,1))
    aa[0] =(xb[2]+xb[0])/2  
    aa[1] =xb[0]  
    aa[2] =xb[1]  
    aa[3] =xb[2]
    aa[4] =(xb[2]+xb[0])/2  
    b= np.zeros((3,1))
    b[0] =xa[0]  
    b[1] =xa[1]  
    b[2] =xa[2]  
    bb= np.zeros((5,1))
    bb[0] =(xa[2]+xa[0])/2  
    bb[1] =xa[0]  
    bb[2] =xa[1]  
    bb[3] =xa[2]
    bb[4] =(xa[2]+xa[0])/2  
    c= np.zeros((3,1))
    c[0] =xt[0]  
    c[1] =xt[1]  
    c[2] =xt[2]  
    cc= np.zeros((5,1))
    cc[0] =(xt[2]+xt[0])/2  
    cc[1] =xt[0]  
    cc[2] =xt[1]  
    cc[3] =xt[2]
    cc[4] =(xt[2]+xt[0])/2  
    la = plt.plot(x,a,'k')
    laa = plt.plot(z,aa,'k--')
    lc = plt.plot(x,c,'g')
    lcc = plt.plot(z,cc,'g--')
    lb = plt.plot(x,b,'r')
    lbb = plt.plot(z,bb,'r--')
    plt.text(3., 1.4, 'Background',verticalalignment='bottom', horizontalalignment='right', color='k')
    plt.text(3., 1.2, 'Analysis',verticalalignment='bottom', horizontalalignment='right', color='r')
    plt.text(3., 1.0, 'Truth',verticalalignment='bottom', horizontalalignment='right', color='g')
    #ll = plt.legend(loc='lower right')
    lx = plt.xlabel('Spatial Grid')
    ly = plt.ylabel('Tracer Concentration')
# tell matplotlib which xticks to plot 
    plt.xticks((1,2,3,))  
# tell matplotlib which yticks to plot 
    plt.yticks((0,1,2,3,4))  
    plt.title('(a) Tracer Distribution ($t=t_0$)')
    plt.grid()
    plt.xlim([0.5, 3.5])
    plt.ylim([0, 4])
    # Initialize minor ticks
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    x_pos = [0.6]
    y_pos = [3.6]
    x_direct = [1.4]
    y_direct = [0]
    q=plt.quiver(x_pos,y_pos,x_direct,y_direct,color=['k'],scale=5)
    p = plt.quiverkey(q,1.9,3.3,1.4,"Background Wind (1.40)",coordinates='data',color='w',labelcolor='k')
    y2_pos = [3.2]
    x2_direct = [1.04]
    q=plt.quiver(x_pos,y2_pos,x2_direct,y_direct,color=['r'],scale=5)
    p = plt.quiverkey(q,1.9,2.9,1.04,"Analyzed Wind (1.04)",coordinates='data',color='w',labelcolor='r')
    y3_pos = [2.8]
    x3_direct = [1.]
    q=plt.quiver(x_pos,y3_pos,x3_direct,y_direct,color=['g'],scale=5)
    p = plt.quiverkey(q,1.85,2.5,1,"True Wind (1.00)",coordinates='data',color='w',labelcolor='g')
    p2 = plt.subplot(2,1,2)
    a= np.zeros((3,1))
    a[0] =xb1[0]  
    a[1] =xb1[1]  
    a[2] =xb1[2]  
    aa= np.zeros((5,1))
    aa[0] =(xb1[2]+xb1[0])/2  
    aa[1] =xb1[0]  
    aa[2] =xb1[1]  
    aa[3] =xb1[2]
    aa[4] =(xb1[2]+xb1[0])/2  
    b= np.zeros((3,1))
    b[0] =xa1[0]  
    b[1] =xa1[1]  
    b[2] =xa1[2]  
    bb= np.zeros((5,1))
    bb[0] =(xa1[2]+xa1[0])/2  
    bb[1] =xa1[0]  
    bb[2] =xa1[1]  
    bb[3] =xa1[2]
    bb[4] =(xa1[2]+xa1[0])/2  
    c= np.zeros((3,1))
    c[0] =xt1[0]  
    c[1] =xt1[1]  
    c[2] =xt1[2]  
    cc= np.zeros((5,1))
    cc[0] =(xt1[2]+xt1[0])/2  
    cc[1] =xt1[0]  
    cc[2] =xt1[1]  
    cc[3] =xt1[2]
    cc[4] =(xt1[2]+xt1[0])/2  
    #la = plt.plot(x,a,'k',label='Background')
    la = plt.plot(x,a,'k')
    laa = plt.plot(z,aa,'k--')
    lc = plt.plot(x,c,'g')
    lcc = plt.plot(z,cc,'g--')
    lb = plt.plot(x,b,'r')
    lbb = plt.plot(z,bb,'r--')
    plt.text(3., 1.4, 'Background',verticalalignment='bottom', horizontalalignment='right', color='k')
    plt.text(3., 1.2, 'Analysis',verticalalignment='bottom', horizontalalignment='right', color='r')
    plt.text(3., 1.0, 'Truth',verticalalignment='bottom', horizontalalignment='right', color='g')
    #ll = plt.legend(loc='lower right')
    lx = plt.xlabel('Spatial Grid')
    ly = plt.ylabel('Tracer Concentration')
# tell matplotlib which xticks to plot 
    plt.xticks((1,2,3))  
# tell matplotlib which yticks to plot 
    plt.yticks((0,1,2,3,4))  
# labelling the yticks according to your list
#    plt.yticklabels(['A','B','C','D'])
    plt.title('(b) Tracer Distribution ($t=t_1$)')
    plt.grid()
    plt.ylim([0, 4])
    # Initialize minor ticks
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.xlim([0.5, 3.5])
    plt.ylim([0, 4])
    plt.savefig('Figure1Project1EHTP2020.png', format='png', dpi=500,transparent=False)
    plt.show()
elif "2" == q:
    u=1
    print('wind\n',np.round(u,4))
    eb = np.zeros((1,1))
    eb[0]=-0.2
    print('Diffusion coefficient background error\n',np.round(eb,4))
    R= np.zeros((3,3))
    R[0][0]=0.01
    R[1][1]=0.01
    R[2][2]=0.01
    print('obs error variance\n',np.round(R,4))
    xt = np.zeros((4,1))
    xt[0]=1
    xt[1]=2
    xt[2]=3
    xt[3]=0.2
    xb = np.zeros((4,1))
    xb[0]=1
    xb[1]=2
    xb[2]=3
    xb[3]=xt[3]+eb[0]
    B= np.zeros((4,4))
    B[0][0]=R[0][0]
    B[1][1]=R[0][0]
    B[2][2]=R[0][0]
    B[3][3]=1
    print('Background error covariance at time t0\n',np.round(B,4))
    HT = np.zeros((4,3))
    HT[0][0]=1
    HT[1][1]=1
    HT[2][2]=1
    H=np.transpose(HT)
    print('Obs. operator \n',np.round(H,4))
    M= np.zeros((4,4))
    M[0][0]=1-2*xb[3]
    M[0][1]=-0.5*u+xb[3]
    M[0][2]=0.5*u+xb[3]
    M[0][3]=0
    M[1][0]=0.5*u+xb[3]
    M[1][1]=1-2*xb[3]
    M[1][2]=-0.5*u+xb[3]
    M[1][3]=0
    M[2][0]=-0.5*u+xb[3]
    M[2][1]=0.5*u+xb[3]
    M[2][2]=1-2*xb[3]
    M[2][3]=0
    M[3][0]=0
    M[3][1]=0
    M[3][2]=0
    M[3][3]=1-2*xb[3]
    print('Model\n',np.round(M,4))
    Mt= np.zeros((4,4))
    Mt[0][0]=1-2*xt[3]
    Mt[0][1]=-0.5*u+xt[3]
    Mt[0][2]=0.5*u+xt[3]
    Mt[0][3]=0
    Mt[1][0]=0.5*u+xt[3]
    Mt[1][1]=1-2*xt[3]
    Mt[1][2]=-0.5*u+xt[3]
    Mt[1][3]=0
    Mt[2][0]=-0.5*u+xt[3]
    Mt[2][1]=0.5*u+xt[3]
    Mt[2][2]=1-2*xt[3]
    Mt[2][3]=0
    Mt[3][0]=0
    Mt[3][1]=0
    Mt[3][2]=0
    Mt[3][3]=1
    print('True model\n',np.round(Mt,4))
    L= np.zeros((4,4))
    L[0][0]=1
    L[0][1]=-0.5*u
    L[0][2]=0.5*u
    L[0][3]=xb[1]-2*xb[0]+xb[2]
    L[1][0]=0.5*u
    L[1][1]=1
    L[1][2]=-0.5*u
    L[1][3]=xb[2]-2*xb[1]+xb[0]
    L[2][0]=-0.5*u
    L[2][1]=0.5*u
    L[2][2]=1
    L[2][3]=xb[0]-2*xb[2]+xb[1]
    L[3][0]=0
    L[3][1]=0
    L[3][2]=0
    L[3][3]=1
    print('Tangent linear model\n',np.round(L,4))
    print('Adjoint model\n',np.round(np.transpose(L),4))
    yo= np.zeros((3,1))
    yo=np.dot(np.dot(H,Mt),xt)
    print('Observation\n',np.round(yo,4))
    d = np.zeros((3,1))
    d=yo-np.dot(np.dot(H,M),xb)
    gamma=np.zeros((3,1))
    gamma[0]=xb[1]-2*xb[0]+xb[2]
    gamma[1]=xb[2]-2*xb[1]+xb[0]
    gamma[2]=xb[0]-2*xb[2]+xb[1]
    #The innovation can also be given by d=-np.dot(eb,gamma)
    print('Innovation\n',np.round(d,4))
    H=np.dot(H,L)
    HT=np.transpose(H)
    BHT=np.dot(B,HT)
    HB=np.dot(H,B)
    HBHT=np.dot(HB,HT)
    K=np.dot(BHT,inv(np.matrix(HBHT+R)))
    xa= np.zeros((4,1))
    xa=xb+np.dot(K,d)
    print('Background at time t0\n',np.round(xb,4))
    print('4D-Var Analysis at time t0\n',np.round(xa,4))
    print('Truth at time t0\n',np.round(xt,4))
    xb1= np.zeros((4,1))
    xb1=np.dot(M,xb)
    xa1= np.zeros((4,1))
    xa1=xb1+np.dot(L,np.dot(K,d))
    print('Background at time t1\n',np.round(xb1,4))
    print('4D-Var Analysis at time t1\n',np.round(xa1,4))
    xt1= np.zeros((4,1))
    xt1=np.dot(Mt,xt)
    print('Truth at time t1\n',np.round(xt1,4))
    kappa=xa[3]
    print('The optimal value of the Nondimensional Diffusion Coefficient\n',np.round(kappa,4))
    k=(kappa*2*math.pi)/3
    print('The optimal value of the Diffusion Coefficient\n',np.round(k,4))
else:
    sys.exit("Project number is incorrect.")
