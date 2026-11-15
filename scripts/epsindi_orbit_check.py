# Expected offset of the Eps Indi Ba+Bb photocentre from the Gaia DR3 (Ba) position propagated with Gaia DR3 PM.
# Orbit: Chen et al. 2022 Table 4 (P, a, e, omega, Omega, i); T0 fitted here to their Table 3 relative astrometry.
import math
P,a,e,om,Om,inc=11.0197,0.66158,0.54042,math.radians(328.27),math.radians(147.959),math.radians(77.082)
B=0.4431  # M_Bb/(M_Ba+M_Bb)
obs=[(2004.730,0.88310,140.317),(2005.423,0.93141,142.648),(2005.997,0.93987,144.356),(2006.863,0.89489,147.111),
     (2007.849,0.73666,151.017),(2008.915,0.39126,160.569),(2009.458,0.14626,186.175),(2010.582,0.32838,332.295),
     (2011.543,0.17352,12.950),(2012.545,0.25518,107.165),(2013.431,0.47861,126.845)]
def rel(t,T0,om=om):
    M=2*math.pi*((t-T0)/P % 1.0); E=M
    for _ in range(100): E=E-(E-e*math.sin(E)-M)/(1-e*math.cos(E))
    X=math.cos(E)-e; Y=math.sqrt(1-e*e)*math.sin(E)
    A=a*(math.cos(om)*math.cos(Om)-math.sin(om)*math.sin(Om)*math.cos(inc))
    Bc=a*(math.cos(om)*math.sin(Om)+math.sin(om)*math.cos(Om)*math.cos(inc))
    F=a*(-math.sin(om)*math.cos(Om)-math.cos(om)*math.sin(Om)*math.cos(inc))
    G=a*(-math.sin(om)*math.sin(Om)+math.cos(om)*math.cos(Om)*math.cos(inc))
    dN=A*X+F*Y; dE=Bc*X+G*Y
    return dE,dN
def chi(T0,om_):
    s=0
    for t,r,th in obs:
        dE,dN=rel(t,T0,om_); oE=r*math.sin(math.radians(th)); oN=r*math.cos(math.radians(th))
        s+=(dE-oE)**2+(dN-oN)**2
    return math.sqrt(s/len(obs))
best=min(((chi(2000+k*0.001,o),2000+k*0.001,o) for k in range(0,11020) for o in (om,om+math.pi)),key=lambda x:x[0])
print("rms residual (arcsec), T0, omega(deg):",round(best[0],4),round(best[1],3),round(math.degrees(best[2])%360,2))
T0=best[1]; om_=best[2]
for t,r,th in obs[:3]+obs[-2:]:
    dE,dN=rel(t,T0,om_); print(t,"model rho,theta",round(math.hypot(dE,dN),4),round(math.degrees(math.atan2(dE,dN))%360,2),"obs",r,th)
dmu=(3987.41-3981.977,-2505.35-(-2466.832))  # mu_bary(Chen22) - mu_GaiaDR3, mas/yr
rE0,rN0=rel(2016.0,T0,om_)
print("r(2016.0) E,N:",round(rE0,3),round(rN0,3))
h=0.25
vE=(rel(2016.0+h,T0,om_)[0]-rel(2016.0-h,T0,om_)[0])/(2*h); vN=(rel(2016.0+h,T0,om_)[1]-rel(2016.0-h,T0,om_)[1])/(2*h)
print("Ba orbital PM about barycentre near 2016 (mas/yr) E,N:",round(-B*vE*1e3,1),round(-B*vN*1e3,1)," | Gaia-minus-barycentre PM observed:",round(-dmu[0],1),round(-dmu[1],1))
bands={'z(0.9um)':0.207,'J':0.332,'H':0.160,'K':0.115,"L'":0.184,"M'":0.416}
for t in (2025.40,2025.90,2026.40):
    rE,rN=rel(t,T0,om_)
    print("t=%.2f  Bb-Ba: rho=%.3f theta=%.1f"%(t,math.hypot(rE,rN),math.degrees(math.atan2(rE,rN))%360))
    for b,beta in bands.items():
        dE=dmu[0]*1e-3*(t-2016.0)+(beta-B)*rE+B*rE0
        dN=dmu[1]*1e-3*(t-2016.0)+(beta-B)*rN+B*rN0
        print("   %-9s beta=%.3f  photocentre minus Gaia-extrapolated: dRA*=%+.2f\"  dDec=%+.2f\""%(b,beta,dE,dN))
# photocentre PM during the SPHEREx year
for b,beta in (('K',0.115),("L'",0.184),("M'",0.416)):
    (e1,n1),(e2,n2)=rel(2025.4,T0,om_),rel(2026.4,T0,om_)
    print("photocentre PM 2025.4-2026.4, %s: %.1f, %.1f mas/yr"%(b,3987.41+(beta-B)*(e2-e1)*1e3,-2505.35+(beta-B)*(n2-n1)*1e3))
