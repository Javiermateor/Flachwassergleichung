import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 100
CFL = 0.9
imageCounter = 0

fig, axs = plt.subplots(2, 1, figsize=(15, 6))
fig.suptitle(f'N = {N} CFL = {CFL}', fontsize=15)
fig.subplots_adjust(hspace=0.5)

def Anfangsbedigungen(Nx,CFL,verbose=True):
    xmax = 10 
    xmin = 0
    dx = (xmax-xmin)/Nx 
    x = np.linspace(xmin, xmax, Nx) 
    dt = CFL*dx                     # Zeitschrittweite
    t_ende = 5                      # Endzeit
    Nt = int(t_ende/dt)             # Anzahl der Zeitschritte
    
    #Upwind-Verfahren
    c_positiv = 1*(dt/dx)
    c_negativ = 0
    
    #Lax-Friedrich / Lax-Wendroff-Verfahren
    c=dt/dx
    
    # Glatte Anfangsbedingung
    Uo_glatt = np.exp(-2.5*(x-2)**2)
    
    # Initialisierung der Zustandsmatrix
    U_glatt = np.zeros((Nt+1, Nx))
    U_glatt[0] = Uo_glatt
    
    # Unstetige Anfangsbedingung
    Uo_unstetig = np.where(np.logical_and(x>=1, x<=3), 1, 0)
    
    # Initialisierung der Zustandsmatrix
    U_unstetig = np.zeros((Nt+1, Nx))
    U_unstetig[0] = Uo_unstetig
    
    if verbose:
        return U_glatt, U_unstetig, c_positiv, c_negativ, Nt, Nx, x, dt #Upwind-Verfahren
    else:
        return U_glatt, U_unstetig, c, Nt, Nx,x,dt #Lax-Friedrich / Lax-Wendroff-Verfahren / Analytische Lösung (verbose=False)
    

def upwind_verfahren(Nx, CFL=1):    
    global imageCounter
    U_glatt, U_unstetig, c_positiv, c_negativ, Nt, Nx, x, dt = Anfangsbedigungen(Nx,CFL)
    
    #Lösung des beschriebenen Problems
    for n in range(Nt):
        for i in range(1, Nx-1):
            U_glatt[n+1, i] = U_glatt[n, i] - c_positiv*(U_glatt[n, i] - U_glatt[n, i-1]) + c_negativ*(U_glatt[n, i+1] - U_glatt[n, i])
            U_unstetig[n+1, i] = U_unstetig[n, i] - c_positiv*(U_unstetig[n, i] - U_unstetig[n, i-1]) + c_negativ*(U_unstetig[n, i+1] - U_unstetig[n, i])
        #Randbedingungen
        U_glatt[n+1, 0] = 0
        U_glatt[n+1, Nx-1] = 0
        U_unstetig[n+1, 0] = 0
        U_unstetig[n+1, Nx-1] = 0
        
        axs[0].cla()
        axs[1].cla()
        axs[0].plot(x, U_glatt[n], label='Upwind')
        axs[1].plot(x, U_unstetig[n], label='Upwind')
        axs[0].set_title(f'Glatte Anfangsbedingung')
        axs[0].set_xlabel('U')
        axs[0].set_ylabel('x')
        axs[1].set_title(f'Unstetige Anfangsbedingung')
        axs[1].set_xlabel('U')
        axs[1].set_ylabel('x')
        axs[0].set_ylim(0, 1.1)
        axs[1].set_ylim(0, 1.1)
        axs[0].legend()
        axs[1].legend()
        plt.draw()
        plt.savefig('pngs/{:03d}'.format(imageCounter), dpi=300)
        imageCounter += 1

    return U_glatt, U_unstetig, x, dt

U_glatt_upwind, U_unstetig_upwind, _, _ = upwind_verfahren(N, CFL)

#Analytische Lösung
def analytisch(Nx, CFL=1):
    global imageCounter
    U_glatt, U_unstetig, c, Nt, Nx, x, dt = Anfangsbedigungen(Nx,CFL,verbose = False)
    t = np.linspace(0, 5, Nt)     
    #Lösung des beschriebenen Problems
    for n in range(Nt):
        for i in range(0, Nx-1):
            U_glatt[n+1, i+1] = np.exp(-2.5*(x[i]-t[n]-2)**2)
            U_unstetig[n+1, i+1] = np.logical_and(x[i]-t[n]>=1, x[i]-t[n]<=3)
        #Randbedingungen
        U_glatt[n+1, 0] = 0
        U_glatt[n+1, -1] = 0
        U_unstetig[n+1, 0] = 0
        U_unstetig[n+1, -1] = 0        

        axs[0].cla()
        axs[1].cla()
        axs[0].plot(x, U_glatt_upwind[-1], label='Upwind')
        axs[1].plot(x, U_unstetig_upwind[-1], label='Upwind')
        axs[0].plot(x, U_glatt[n], label='Analytisch')
        axs[1].plot(x, U_unstetig[n], label='Analytisch')
        axs[0].set_title(f'Glatte Anfangsbedingung')
        axs[0].set_xlabel('U')
        axs[0].set_ylabel('x')
        axs[1].set_title(f'Unstetige Anfangsbedingung')
        axs[1].set_xlabel('U')
        axs[1].set_ylabel('x')
        axs[0].set_ylim(0, 1.1)
        axs[1].set_ylim(0, 1.1)
        axs[0].legend()
        axs[1].legend()
        plt.draw()
        plt.savefig('pngs/{:03d}'.format(imageCounter), dpi=300)
        imageCounter += 1

    return U_glatt, U_unstetig, x, dt

analytisch(N, CFL)

