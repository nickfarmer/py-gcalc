#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:28:58 2018

@author: nick
"""

# function to read ds6 datafile into pandas dataframe

import pandas as pd
import numpy as np
import math
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def ds6import(filename):
    """Import a datafile in the format used by Holland & Powell for the 6x series of datafiles located at filename, and return a pandas dataframe with appropriately labelled rows and columns.""" 
    hpdata=[]
    names=[]
    atoms=[]

    with open(filename) as datafile:
        lines=datafile.readlines()
        for i in range(0, len(lines)-3):
            nameline=lines[i]
            stpline=lines[i+1]
            cpline=lines[i+2]
            akline=lines[i+3]
            if len(nameline) > 8 and nameline[7].isalpha():
                nameline=nameline.split()
                atoms=[]
                for j in range(3,len(nameline)):
                    if j % 2:
                        atoms.append(float(nameline[j]))
                        #nameline=nameline[0]
                #names.append(nameline)
                stpline=stpline.split()
                #stpline=stpline[0],stpline[1],stpline[2]
                #stp.append(stpline)
                cpline=cpline.split()
                #cp.append(cpline)
                akline=akline.split()
                if len(akline)==8:
                    completeline=(sum(atoms),stpline[0],stpline[1],stpline[2],cpline[0],cpline[1],cpline[2],cpline[3],akline[0],akline[1],akline[2],akline[3],akline[4],akline[5],akline[6],akline[7])
                else:
                    completeline=(sum(atoms),stpline[0],stpline[1],stpline[2],cpline[0],cpline[1],cpline[2],cpline[3],akline[0],akline[1],akline[2],akline[3],akline[4])   
                hpdata.append(completeline)
                names.append(nameline[0])
                #ak.append(akline)
            
    index = pd.Index(names, name='rows')
    columns = pd.Index(["atoms","H0","S0","V0","CpA","CpB","CpC","CpD","alpha0","k0","kprime0","kdprime0", "Landau", "Tc", "Smax", "Vmax"], name='cols')
    hpdata = pd.DataFrame(hpdata, index=index, columns=columns)
    return hpdata

# functions to perform calculations based on methodology of Holland & Powell (2011)
    
def HtminusH(phase,T,hpdata):
    """For phase, calulate the difference between standard state H and H at temperature T, using dataset hpdata.""" 
    return float(hpdata.loc[phase,'CpA'])*(T-298.15)+0.5*float(hpdata.loc[phase,'CpB'])*(T*T-298.15*298.15)-float(hpdata.loc[phase,'CpC'])*(1/T-1/298.15)+2*float(hpdata.loc[phase,'CpD'])*(math.sqrt(T)-math.sqrt(298.15))

def StminusS(phase,T,hpdata):
    """For phase, calulate the difference between standard state S and S at temperature T, using dataset hpdata."""
    return 1000*(float(hpdata.loc[phase,'CpA'])*math.log(T/298.15)+float(hpdata.loc[phase,'CpB'])*(T-298.15)-0.5*float(hpdata.loc[phase,'CpC'])*(1/(T*T)-1/(298.15*298.15))-2*float(hpdata.loc[phase,'CpD'])*(1/math.sqrt(T)-1/math.sqrt(298.15)))

def GTP0(phase,T,hpdata):
    """For phase, calulate the difference between standard state G and G at temperature T, using dataset hpdata."""
    return ((float(hpdata.loc[phase,'H0'])+HtminusH(phase,T,hpdata))*1000)-(((float(hpdata.loc[phase,'S0']))*1000)+StminusS(phase,T,hpdata))*T

def EOSa(phase,P,hpdata):
    """Calculate the EOS a parameter for phase at pressure P, using dataset hpdata. HP11 eq. 3"""
    return (1+float(hpdata.loc[phase, 'kprime0']))/((1+float(hpdata.loc[phase, 'kprime0'])+(float(hpdata.loc[phase, 'k0'])*float(hpdata.loc[phase, 'kdprime0']))))

def EOSb(phase,P,hpdata):
    """Calculate the EOS b parameter for phase at pressure P, using dataset hpdata. HP11 eq. 3"""
    return (float(hpdata.loc[phase, 'kprime0'])/float(hpdata.loc[phase, 'k0']))-float(hpdata.loc[phase, 'kdprime0'])/(1+float(hpdata.loc[phase, 'kprime0']))

def EOSc(phase,P,hpdata):
    """Calculate the EOS c parameter for phase at pressure P, using dataset hpdata. HP11 eq. 3"""
    return (1+float(hpdata.loc[phase, 'kprime0'])+float(hpdata.loc[phase, 'k0'])*float(hpdata.loc[phase, 'kdprime0']))/(float(hpdata.loc[phase, 'kprime0'])*float(hpdata.loc[phase, 'kprime0'])+float(hpdata.loc[phase, 'kprime0'])-float(hpdata.loc[phase, 'k0'])*float(hpdata.loc[phase, 'kdprime0']))

def Pth(phase,T,hpdata):
    """Calculate thermal pressure Pth for phase at temperature T, using dataset hpdata. HP11 eq. 11"""
    theta=10636/(((float(hpdata.loc[phase, 'S0'])*1000)/float(hpdata.loc[phase, 'atoms']))+6.44)
    xi0=(theta/298.15)*(theta/298.15)*math.exp(theta/298.15)/((math.exp(theta/298.15)-1)*(math.exp(theta/298.15)-1))
    Pthermal=float(hpdata.loc[phase, 'alpha0'])*float(hpdata.loc[phase, 'k0'])*theta/xi0*(1/(math.exp(theta/T)-1)-1/(math.exp(theta/298.15)-1))
    return(Pthermal)

def V(phase,P,T,hpdata):
    """Calculate volume V for phase at pressure P and temperature T, using dataset hpdata. HP11 eq. 12"""
    return float(hpdata.loc[phase, 'V0'])*(1-EOSa(phase,P,hpdata)*(1-math.pow((1+EOSb(phase,P,hpdata)*(P-Pth(phase,T))),(-EOSc(phase,P,hpdata)))))

def PVcorr(phase,P,T,hpdata):
    """Calculate term in brackets in HP11 eq. 13 (PVcorr) for phase at pressure P, temperature T, using dataset hpdata."""
    return 1-EOSa(phase,P,hpdata)+(EOSa(phase,P,hpdata)*(math.pow((1-EOSb(phase,P,hpdata)*Pth(phase,T,hpdata)),(1-EOSc(phase,P,hpdata)))-math.pow((1+EOSb(phase,P,hpdata)*(P-Pth(phase,T,hpdata))),(1-EOSc(phase,P,hpdata))))/(EOSb(phase,P,hpdata)*(EOSc(phase,P,hpdata)-1)*P))

def GPTminusGP0T(phase,P,T,hpdata):
    """Calculate difference between G for phase at pressure P and temperature T, and G at temperature T and pressure 1 bar. HP11 eq. 13"""
    return PVcorr(phase,P,T,hpdata)*P*float(hpdata.loc[phase, 'V0'])

def GPT(phase,P,T,hpdata):
    """Calculate G for phase at pressure P and temperature T, using dataset hpdata. If applicable, calls function to calculate contribution of 2nd order phase transition (Landau)"""
    if hpdata.loc[phase,'Landau']=="1":
        return GTP0(phase,T,hpdata)+GPTminusGP0T(phase,P,T,hpdata)*1000+Glandau(phase,P,T,hpdata)
    else:
        return GTP0(phase,T,hpdata)+GPTminusGP0T(phase,P,T,hpdata)*1000
        
    
def GPTnoLandau(phase,P,T,hpdata):
    """Calculate G for phase at pressure P and temperature T, using dataset hpdata. Ignores Landau. Useful for testing purposes."""
    return GTP0(phase,T,hpdata)+GPTminusGP0T(phase,P,T,hpdata)*1000

def Glandau(phase,P,T,hpdata):
    """Calculates contribution to G for phase at pressure P and temperature T of 2nd order phase transition modelled using Landau theory, using dataset hpdata."""
    # retrieve relevant data from dataframe
    Tc=float(hpdata.loc[phase,'Tc'])
    Vmax=float(hpdata.loc[phase,'Vmax'])
    Smax=float(hpdata.loc[phase,'Smax'])
    # define T*c in terms of Vmax, Smax and P
    Tsc=Tc+(Vmax/Smax)*P
    # if T above critical T, Q=0
    if T<Tsc:
        Q=math.pow(((Tsc-T)/Tc),0.25)
    else:
        Q=0
    Q298=math.pow(((Tc-298.15)/Tc),0.25)
    return (Tc*Smax*(Q298**2-(1/3)*Q298**6)-Smax*(Tsc*Q**2-(1/3)*Tc*Q**6)-T*(Smax*(Q298**2-Q**2))+P*(Vmax*Q298**2))*1000

# functions to find the pressure or temperature of univariant reactions
# change these to be able to model reactions with more than 1 product or reactant

def univariantPcalc(P,T,reactants,products,hpdata):
    """Return DeltaG across a reaction reactants=products at P,T."""
    prodG=[]
    reacG=[]
    for i in range(0,len(products)):
        prodG.append(GPT(products[i],P,T,hpdata))
    for i in range(0,len(reactants)):
        reacG.append(GPT(reactants[i],P,T,hpdata))
    return sum(prodG)-sum(reacG)
    
def univariantPseek(Pguess,T,reactants,products,hpdata):
    """Return the pressure of a univariant reaction reactants=products at temperature T, using dataset hpdata."""
    return optimize.newton(univariantPcalc, x0=Pguess, args=(T, reactants, products, hpdata))

def univariantTcalc(T,P,reactants,products,hpdata):
    """Return DeltaG across a reaction reactants=products at T,P. Similar to univariantPcalc, but tales arguments in different order."""
    prodG=[]
    reacG=[]
    for i in range(0,len(products)):
        prodG.append(GPT(products[i],P,T,hpdata))
    for i in range(0,len(reactants)):
        reacG.append(GPT(reactants[i],P,T,hpdata))
    return sum(prodG)-sum(reacG)

def univariantTseek(Tguess,P,reactants,products,hpdata):
    """Return the pressure of a univariant reaction reactants=products at temperature T, using dataset hpdata."""
    return optimize.newton(univariantTcalc, x0=Tguess, args=(P, reactants, products, hpdata))

# find invariant point, eg. ky=and=sill

def invariantPTsumsq(PTguess,phase1,phase2,phase3,hpdata):
    """Return the sum of squares of Delta G of reactions between each of 3 phases (phase1, phase2, phase3) at PT conditions PTguess using dataset hpdata"""
    P,T=PTguess
    eq1=GPT(phase1,P,T,hpdata)-GPT(phase2,P,T,hpdata)
    eq2=GPT(phase2,P,T,hpdata)-GPT(phase3,P,T,hpdata)
    eq3=GPT(phase1,P,T,hpdata)-GPT(phase3,P,T,hpdata)
    return eq1*eq1+eq2*eq2+eq3*eq3
    
def invariantPTseek(PTguess,phase1,phase2,phase3,hpdata):
    """Minimize the sum of squares (invariantPTsumsq) to return the P,T of the invariant point at which phase1, phase2, phase3 are stable, using dataset hpdata."""
    return optimize.minimize(invariantPTsumsq, PTguess, args=(phase1, phase2, phase3, hpdata), method='Nelder-Mead')

def univariantTrangeC(Tmin,Tmax,Pguess,reactants,products,hpdata):
    """Return values for P of univariant phase transition every 10 in T range (in C) Tmin-Tmax, with a starting guess for pressure Pguess, using dataset hpdata.""" 
    Trange=[]
    Trange=np.arange(Tmin,Tmax,10)
    TrangeK=Trange+273.15
    modelPcurve=[]
    for i in range(0,len(TrangeK)):
        modelPcurve.append(univariantPseek(30,TrangeK[i],reactants,products,hpdata))
    return modelPcurve

def plotunivariantTrangeC(Tmin,Tmax,Pguess,reactants,products,hpdata):
    """Plot values for P of univariant phase transition every 10 in T range (in C) Tmin-Tmax, with a starting guess for pressure Pguess, using dataset hpdata.""" 
    Trange=[]
    Trange=np.arange(Tmin,Tmax,10)
    ax=plt.plot(Trange,univariantTrangeC(Tmin,Tmax,Pguess,reactants,products,hpdata),'b-',label="This model")
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (kbar)')
    return ax