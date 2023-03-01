import numpy as np
import matplotlib.pyplot as plt
g = 9.81
from IPython import display

#Randbedingungen Funktionen

def periodischer_block(h):
    h[0,:] = h[-2,:]
    h[-1,:] = h[1,:]
    h[:,0] = h[:,1] # Reflektierend
    h[:,-1] = h[:,-2] # reflektierend
    return h

def reflektierender_block(h):
    h[0,:] = h[1,:]
    h[-1,:] = h[-2,:]
    h[:,0] = h[:,1]
    h[:,-1] = h[:,-2]
    return h

#Anfangsbedingungen Funktionen für Aufgaben 3.2 und 3.3

def anfangsbedingungen32(hh, ht, Nx, Ny):
    x = np.linspace(0, 10, Nx)
    y = np.linspace(0, 10, Ny)
    # Initialisierung der Arrays
    h = np.zeros((Nx, Ny), dtype=np.double)
    hu = np.zeros((Nx, Ny), dtype=np.double)
    hv = np.zeros((Nx, Ny), dtype=np.double)

    for j in range(Nx):
        for k in range(Ny):
            if (4 <= x[j] <= 6) and (4 <= y[k] <= 6):
                h[j, k] = hh
            else:
                h[j, k] = ht
    return h, hu, hv

def anfangsbedingungen33(Nx,Ny,darstellung=1):
    
     # Anfangsparameter
    interval = 100e3
    
    dx = interval
    dy = dx

    x = np.arange(0, interval * Nx, interval)
    y = np.arange(0, interval * Ny, interval)

    [Y, X] = np.meshgrid(y, x)
    
    # Konstanten 
    Ωe = 7.2921e-5 # Drehfrequenz der Erde in 1/s
    Re = 6371e3 # Erdradius in m
    y0 = 3e6 # Breitengrad in Grad
    
    #Berechnung des Coriolisfaktors 
    
    latitude = 35
    θ_0 = np.deg2rad(latitude) # Breitengrad in Bogenmaß
    fc_0 = 2*Ωe*np.sin(θ_0) # Mittlere Zentrifugalkraft in 1/s^2
    f = fc_0 + (2*Ωe/Re)*(Y-y0) # Coriolisfaktor
    
    
    if darstellung == 1:
        #Barotropische Instabilität
        W = 10000 - 500 * np.tanh(3e-6 * (Y - y0))
        R = np.random.randint(-1, 6, size=W.shape)
        W = W+R
        B = np.zeros((Nx, Ny), dtype=np.double)
        
    elif darstellung==2:   
        #Rossby Wellen in der nördlichen Hemisphäre
        westwind = 30
        W = 10000 + (westwind/g)*fc_0*(Y-y0)
        
        sigma_x = 10*dx
        sigma_y = 10*dy
        
        B = 2000*np.exp(-0.5*((X-12000000)/sigma_x)**2-0.5*((Y-y0)/sigma_y)**2) #https://en.wikipedia.org/wiki/Gaussian_function
    
    [dWdx, dWdy] = np.gradient(W, *[dy, dx])
    [dBdx, dBdy] = np.gradient(B, *[dy, dx])
    
    u = (-g/f)*dWdy
    v = (g/f)*dWdx

    print(dWdx)
    print("--------------")
    print(dWdy)
    
    # Initialisierung der Arrays
    
    h = W-B
    hu = h*u
    hv = h*v 
    
    if darstellung == 1:
        return h, hu, hv, f
    elif(darstellung == 2):
        return h, hu, hv, f, dBdx, dBdy
    

def erhaltungsschema_2D(h, hu, hv, CFL, Nx, Ny, darstellung):
    
    #Diskretisierung des Gebiets
    x = np.linspace(0, 10, Nx)
    y = np.linspace(0, 10, Ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Zeitparameter
    z = 0 
    tmax = 5

    # Initialisierung der Matrizen F_j12a, F_j12b, F_j12c und G_k12a, G_k12b, G_k12c
    
    F_j12a = np.zeros((Nx-1, Ny), dtype = np.double)
    F_j12b = np.zeros((Nx-1, Ny), dtype = np.double)
    F_j12c = np.zeros((Nx-1, Ny), dtype = np.double)
    G_k12a = np.zeros((Nx, Ny-1), dtype = np.double)
    G_k12b = np.zeros((Nx, Ny-1), dtype = np.double)
    G_k12c = np.zeros((Nx, Ny-1), dtype = np.double)
    Fa = np.zeros((Nx, Ny), dtype = np.double)
    Fb = np.zeros((Nx, Ny), dtype = np.double)
    Fc = np.zeros((Nx, Ny), dtype = np.double)
    Ga = np.zeros((Nx, Ny), dtype = np.double)
    Gb = np.zeros((Nx, Ny), dtype = np.double)
    Gc = np.zeros((Nx, Ny), dtype = np.double)
    v1 = np.amax(h)
    t1 = np.zeros(1)
   

    #Initialisierung des Plots
    if darstellung == 3:
        fig = plt.figure(figsize=(10,10))
    if darstellung == 2:
        fig = plt.figure(figsize=(20,10))
        ax_contour = fig.add_subplot(111,frameon = False)
        plt.show(block= False)

    # Lax-Friedrichs-Verfahren
    while z < tmax:
        # Berechnung der Eigenwerte
        EWX = np.array([hu[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hu[0,0]/h[0,0]+np.sqrt(g*h[0,0])]) # Quelle: S.34 (3.5)
        EWY = np.array([hv[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hv[0,0]/h[0,0]+np.sqrt(g*h[0,0])])
        for j in range(0,Nx):
            for k in range(0,Ny):
                EWX = np.append(EWX,[hu[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hu[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
                EWY = np.append(EWY,[hv[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hv[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
        dt = CFL * min(dx,dy)/(max(np.amax(EWX), np.amax(EWY))) # Quelle: S. 13 (1.58)
        z += dt
    
        # Fluss in der Mitte
        for j in range(0, Nx):
            for k in range(0, Ny):
                Fa[j,k] = hu[j,k] # Quelle: S.4 (1.1)
                Fb[j,k] = (hu[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)
                Fc[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                Ga[j,k] = hv[j,k] # Quelle: S.4 (1.1)
                Gb[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                Gc[j,k] = (hv[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)

        # Berechnung von F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c
        for j in range(0, Nx-1):
            for k in range(0, Ny-1):
                F_j12a[j,k] = 0.25 * (dx/dt)*(h[j,k]  -   h[j+1,k]) + 0.5 * (Fa[j,k] + Fa[j+1,k]) # Quelle: S.15 (1.63)
                F_j12b[j,k] = 0.25 * (dx/dt)*(hu[j,k] - hu[j+1,k]) + 0.5 * (Fb[j,k] + Fb[j+1,k])
                F_j12c[j,k] = 0.25 * (dx/dt)*(hv[j,k] - hv[j+1,k]) + 0.5 * (Fc[j,k] + Fc[j+1,k])
                G_k12a[j,k] = 0.25 * (dy/dt)*(h[j,k]  -  h[j,k+1]) + 0.5 * (Ga[j,k] + Ga[j,k+1]) # Quelle: S.15 (1.64)
                G_k12b[j,k] = 0.25 * (dy/dt)*(hu[j,k] - hu[j,k+1]) + 0.5 * (Gb[j,k] + Gb[j,k+1])
                G_k12c[j,k] = 0.25 * (dy/dt)*(hv[j,k] - hv[j,k+1]) + 0.5 * (Gc[j,k] + Gc[j,k+1])

        # Berechnung der h, hu und hv
        for j in range(1, Nx-1):
            for k in range(1, Ny-1):
                h[j,k]  = h[j,k]  - (dt/dx) * (F_j12a[j,k] - F_j12a[j-1,k]) - ((dt/dy) * (G_k12a[j,k] - G_k12a[j,k-1])) # Quelle: HA 3 (3.33)
                hu[j,k] = hu[j,k] - (dt/dx) * (F_j12b[j,k] - F_j12b[j-1,k]) - ((dt/dy) * (G_k12b[j,k] - G_k12b[j,k-1])) # + dt * S(U) für 3.3
                hv[j,k] = hv[j,k] - (dt/dx) * (F_j12c[j,k] - F_j12c[j-1,k]) - ((dt/dy) * (G_k12c[j,k] - G_k12c[j,k-1]))

        #Randbedingungen
        h = reflektierender_block(h)
        hu = reflektierender_block(hu)
        hv = reflektierender_block(hv)
        
        u = hu/h
        v = hv/h
        v1 = np.append(v1, [np.amax(h)])
        t1 = np.append(t1, [z])
     
        # Meshgrid für die Darstellung
        X,Y = np.meshgrid(x,y)

        #Surface plot in 3D
        if darstellung == 3:
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, h, cmap='cool', linewidth=0, antialiased=False)
            ax.set_title('Lax-Friedrich')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('h')
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.set_zlim(1.4,2.1)

            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.pause(0.01)
            plt.clf()

        #Contour plot und Quiver plot in 2D
        if darstellung == 2:
            ax_contour.cla()
            ax_contour.set_title('Höhenverlauf')
            contour = ax_contour.contourf(X, Y, h, shading='auto', vmax=2, vmin=1.5, cmap='jet')
            cb = fig.colorbar(contour, ax=ax_contour)
            # ax_cotour.pcolormesh(X, Y, h, shading='auto', vmax=2, vmin=1.5, cmap ='jet')

            ax_contour.quiver(X, Y, v, u)
            ax_contour.set_aspect('equal')

            plt.draw()
            plt.pause(0.01)
            cb.remove()

        
    return h, hu, hv, v1, t1

def maccormack(h, hu, hv, f, CFL, Nx, Ny, darstellung, aufgabe,teil,dBdx=0, dBdy=0):
    
    #Diskretisierung des Gebietes
    if (aufgabe == 3.2):
        x = np.linspace(0, 10, Nx)
        y = np.linspace(0, 10, Ny)
    elif(aufgabe == 3.3):
        interval = 100e3
        x = np.arange(0, interval * Nx, interval)
        y = np.arange(0, interval * Ny, interval)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Zeitparameter
    z = 0
    if aufgabe == 3.2:
        tmax = 5
    if aufgabe == 3.3:
        tmax = 60 * 3600


    # Initialisierung der Matrizen F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c

    Fa = np.zeros((Nx, Ny), dtype = np.double)
    Fb = np.zeros((Nx, Ny), dtype = np.double)
    Fc = np.zeros((Nx, Ny), dtype = np.double)
    Ga = np.zeros((Nx, Ny), dtype = np.double)
    Gb = np.zeros((Nx, Ny), dtype = np.double)
    Gc = np.zeros((Nx, Ny), dtype = np.double)
    Sb = np.zeros((Nx, Ny), dtype = np.double)  
    Sc = np.zeros((Nx, Ny), dtype = np.double)
    h_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    hu_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    hv_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fa_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fb_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fc_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Ga_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Gb_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Gc_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Sb_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Sc_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    v2 = np.zeros(1)
    t2 = np.zeros(1)
    v2[0] =  np.amax(h)

    #Initialisierung des Plots
    
    if darstellung == 3:
        fig = plt.figure(figsize=(10,10))
    if darstellung == 2:
        fig = plt.figure(figsize=(20,10))
        ax_contour = fig.add_subplot(111, frameon=False)
        plt.show(block= False)

    i = 0
    #MacCormack Verfahren 
    while z < tmax:

        
        # Berechnung der Eigenwerte 
        EWX = np.array([hu[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hu[0,0]/h[0,0]+np.sqrt(g*h[0,0])]) # Quelle: S.34 (3.5)
        EWY = np.array([hv[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hv[0,0]/h[0,0]+np.sqrt(g*h[0,0])])
        for j in range(0,Nx):
            for k in range(0,Ny):
                EWX = np.append(EWX,[hu[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hu[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
                EWY = np.append(EWY,[hv[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hv[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
        dt = CFL * min(dx,dy)/(max(np.amax(EWX), np.amax(EWY))) # Quelle: S. 13 (1.58)
        z += dt
        

        # Berechnung von F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c (Flussvektoren)
        for j in range(0, Nx):
            for k in range(0, Ny):
                
                Fa[j,k] = hu[j,k] # Quelle: S.4 (1.1)
                Fb[j,k] = (hu[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)
                Fc[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                
                # Fa[j,k] = h[j,k]*(hu[j,k]/h[j,k])   
                # Fb[j,k] = hu[j,k]*(hu[j,k]/h[j,k])**2 + 0.5*g*(h[j,k]**2)
                # Fc[j,k] = h[j,k]*(hu[j,k]/h[j,k])*(hv[j,k]/h[j,k]) 
                
                Ga[j,k] = hv[j,k] # Quelle: S.4 (1.1)
                Gb[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                Gc[j,k] = (hv[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)
                
                # Ga[j,k] = h[j,k]*(hv[j,k]/h[j,k])
                # Gb[j,k] = h[j,k]*(hu[j,k]/h[j,k])*(hv[j,k]/h[j,k]) 
                # Gc[j,k] = h[j,k]*(hv[j,k]/h[j,k])**2 + 0.5*g*(h[j,k]**2)
                
                if teil == 1:
                    Sb[j,k] =  (f[j,k] * hv[j,k])
                    Sc[j,k] = -(f[j,k] * hu[j,k])
                if teil == 2:
                    Sb[j,k] = -g*(h[j,k])*dBdx[j,k] + (f[j,k] * h[j,k] * hv[j,k]/h[j,k])
                    Sc[j,k] = -g*(h[j,k])*dBdy[j,k] - (f[j,k] *  h[j,k] * hu[j,k]/h[j,k])
        
        
        for j in range(0, Nx-1):
            for k in range(0, Ny-1):
                h_12[j,k]  = h[j,k]   - (dt/dx) * (Fa[j+1,k] - Fa[j,k]) - ((dt/dy) * (Ga[j,k+1] - Ga[j,k])) # Quelle: TUT
                hu_12[j,k] = hu[j,k]  - (dt/dx) * (Fb[j+1,k] - Fb[j,k]) - ((dt/dy) * (Gb[j,k+1] - Gb[j,k])) + (dt * Sb[j, k])
                hv_12[j,k] = hv[j,k]  - (dt/dx) * (Fc[j+1,k] - Fc[j,k]) - ((dt/dy) * (Gc[j,k+1] - Gc[j,k])) + (dt * Sc[j, k])
                
                Fa_12[j,k] = hu_12[j,k] # Quelle: S.4 (1.1)
                Fb_12[j,k] = (hu_12[j,k]**2)/(h_12[j,k]) + 0.5*g*(h_12[j,k]**2)
                Fc_12[j,k] = (hu_12[j,k]*hv_12[j,k])/(h_12[j,k])
                
                Ga_12[j,k] = hv_12[j,k] # Quelle: S.4 (1.1)
                Gb_12[j,k] = (hu_12[j,k]*hv_12[j,k])/(h_12[j,k])
                Gc_12[j,k] = (hv_12[j,k]**2)/(h_12[j,k]) + 0.5*g*(h_12[j,k]**2)
                
                if teil == 1:
                    Sb_12[j,k] = (f[j,k] * hv_12[j,k]) # Quelle: Aufgabestellung
                    Sc_12[j,k] = -(f[j,k] * hu_12[j,k])
                if teil == 2:
                    Sb_12[j,k] = -g*(h_12[j,k])*dBdx[j,k] + (f[j,k] * h_12[j,k] * hv_12[j,k]/h_12[j,k])
                    Sc_12[j,k] = -g*([j,k]*h[j,k])*dBdy[j,k] - (f[j,k] *  h_12[j,k] * hu_12[j,k]/h_12[j,k])
        

        # Berechnung der h, hu und hv
        for j in range(1, Nx-1):
            for k in range(1, Ny-1):
                h[j,k]   = 0.5 * (h[j,k]  +  h_12[j,k])  - (0.5 * (dt/dx) * (Fa_12[j,k] - Fa_12[j-1,k])) - (0.5*(dt/dy) * (Ga_12[j,k] - Ga_12[j,k-1])) # Quelle: TUT
                hu[j,k]  = 0.5 * (hu[j,k] + hu_12[j,k])  - (0.5 * (dt/dx) * (Fb_12[j,k] - Fb_12[j-1,k])) - (0.5*(dt/dy) * (Gb_12[j,k] - Gb_12[j,k-1])) + (dt*0.5*Sb_12[j, k])
                hv[j,k]  = 0.5 * (hv[j,k] + hv_12[j,k])  - (0.5 * (dt/dx) * (Fc_12[j,k] - Fc_12[j-1,k])) - (0.5*(dt/dy) * (Gc_12[j,k] - Gc_12[j,k-1])) + (dt*0.5*Sc_12[j, k])

        #Randbedingungen je nach Aufgabe
        if aufgabe == 3.2:
            h = reflektierender_block(h)
            hu = reflektierender_block(hu)
            hv = reflektierender_block(hv)
            
        if aufgabe == 3.3:
            h = periodischer_block(h)
            hu = periodischer_block(hu)
            hv = periodischer_block(hv)

        print("Schritt: ", i)
        print(f'Zeitschritt {dt} | Zeitschritt Tutorium = 140.2113615804145') 
        
        print('------------------------------------------------')
        print(f'Flussvektoren nach {i} Zeitschritt:')
        print(f'Fa:\n{Fa}')
        print(f'Fb:\n{Fb}')
        print(f'Fc:\n{Fc}')
        print(f'Ga:\n{Ga}')
        print(f'Gb:\n{Gb}')
        print(f'Gc:\n{Gc}')
        
        print('------------------------------------------------')
        print(f'Numerische Höhe und Geschwindigkeiten nach {i} Zeitschritt:')
        print(f'h_12: \n{h}')
        print(f'hu_12: \n{hu}')
        print(f'hv_12: \n{hv}')
        print("")
        
        print('-------------------------------------------------')
    
        print(f'Höhe und Geschwindigkeiten nach {i} Zeitschritt:')
        print(f'h: \n{h}')
        print(f'hu: \n{hu}')
        print(f'hv: \n{hv}')
        print("")

        u = hu/h
        v = hv/h
        v2 = np.append(v2, [np.amax(h)])
        t2 = np.append(v2, [z])

        # Meshgrid für die Darstellung
        if (aufgabe==3.2):
            X,Y = np.meshgrid(x,y)
        elif (aufgabe==3.3):
            Y,X = np.meshgrid(y,x)

        #Surface plot in 3D
        if darstellung == 3:
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, h, cmap='cool', linewidth=0, antialiased=False)
            ax.set_title('Mccormack')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('h')
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.set_zlim(1.4,2.1)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.pause(0.01)
            plt.clf()
        
        #Contour plot und Quiver plot in 2D
        if darstellung == 2:
            ax_contour.cla()
            ax_contour.set_title('Höhenverlauf')
            # ax_contour.set_xlim(0, 240e4)
            # ax_contour.set_ylim(0, 600e4)

            contour = ax_contour.contourf(X, Y, h, vmin=9.5e3, vmax=10.5e3 , shading='auto', cmap='jet')
            cb = fig.colorbar(contour, ax=ax_contour)
            # ax_cotour.pcolormesh(X, Y, h, shading='auto', vmax=2, vmin=1.5, cmap ='jet')
            ax_contour.quiver(X, Y, u, v)
            
            
            #Plot Einstellungen
            # ax_contour.set_xticks(np.arange(0, Nx*1e5, 20e4))
            # ax_contour.set_xticklabels(np.arange(0, Nx, 2))
            # ax_contour.set_xlabel(f' x [{10}\u00B3 km]')
            # ax_contour.set_yticklabels(())
            
            plt.draw()
            plt.pause(0.01)
            cb.remove()
            
            i+=1
    return h, hu, hv, v2, t2

if __name__ == "__main__":
    
    plt.style.use('seaborn')
    
    # # Aufagabe 3.2
    
    # # Lax-Friedrich
    # h, hu, hv = anfangsbedingungen32(hh=2, ht=1.5, Nx=50, Ny=50)
    # h, hu, hv, v1, t1 = erhaltungsschema_2D(h, hu, hv, CFL=0.4, Nx = 50, Ny = 50, darstellung=3)
    
    # #Maccormack
    # h, hu, hv = anfangsbedingungen32(hh = 2, ht = 1.5, Nx = 50, Ny = 50)
    # f = np.zeros([50, 50])
    # h, hu, hv, v2, t2 = maccormack(h, hu, hv, f, CFL=0.4, Nx = 50, Ny = 50,  darstellung= 3, aufgabe= 3.2)
    
    # # Vergleich der Lösungen
    # plt.plot(t2, v2, label='MacCormack')
    # plt.plot(t1, v1, label='Lax-Friedrich')
    # plt.title('MacCormack vs Lax-Friedrich')
    # plt.xlabel('Zeit (s)')
    # plt.ylabel('Höhe (m)')
    # plt.legend()
    # plt.show()
    
    # Aufgabe 3.3
    
    Nx,Ny = 24, 60
    CFL = 0.45
    
    # # 3.1: Barotropische Instabilität
    
    h, hu, hv, f = anfangsbedingungen33(Nx,Ny)
    np.set_printoptions(precision=16)
    print(f)
    print("--------------")
    print(h)
    print("--------------")
    print(hu)
    print("--------------")
    print(hv)
    print("--------------")
    print("--------------")
    
    # # print(f'h: {h.shape} \n{h}\n ')
    # # print(f'hu: {hu.shape} \n{hu}\n ')
    # # print(f'hv: {hv.shape} \n{hv}\n ')
    # # print(f'f:{f.shape} \n{f} ')
    
    h, hu, hv, v3, t3 = maccormack(h, hu, hv, f, CFL, Nx, Ny,  darstellung= 2, aufgabe= 3.3, teil=1)
    
    # 3.2: Rossby Wellen in der nördlichen Hemisphäre
    # h, hu, hv, f, dBdx, dBdy = anfangsbedingungen33(Nx,Ny,darstellung = 2)
    # h, hu, hv, v4, t4 = maccormack(h, hu, hv, f, CFL, Nx, Ny,  darstellung = 2, aufgabe = 3.3, teil = 2, dBdx = dBdx, dBdy = dBdy)

