import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from IPython import display

# Reflektierender Rand um einen Block:

def periodischer_block(h):
    h[:,0] = h[:,-2]
    h[:,-1] = h[:,1]
    h[0,:] = h[-2,:]
    h[-1,:] = h[1,:]
    return h

def reflektierender_block(h):
    h[:,0] = h[:,1]
    h[:,-1] = h[:,-2]
    h[0,:] = h[1,:]
    h[-1,:] = h[-2,:]
    return h

def anfangsbedingungen32(hh, ht, Nx, Ny):
    x = np.linspace(0, 10, Nx)
    y = np.linspace(0, 10, Ny)
    # Initialisierung der Arrays
    h = np.zeros((Nx, Ny), dtype=np.double)
    hu = np.zeros((Nx, Ny), dtype=np.double)
    hv = np.zeros((Nx, Ny), dtype=np.double)

    # Anfangsbedingungen
    # hv und hu sind 0, da u und v 0 sind
    for j in range(Nx):
        for k in range(Ny):
            if (4 <= x[j] <= 6) and (4 <= y[k] <= 6):
                h[j, k] = hh
            else:
                h[j, k] = ht
    return h, hu, hv

def anfangsbedingungen33(darstellung=1):
     # Anfangsparameter
    interval = 100e3
    Nx = 24
    Ny = 60
    
    dx = interval
    dy = dx

    x = np.arange(0, interval * Nx, interval)
    y = np.arange(0, interval * Ny, interval)

    [Y, X] = np.meshgrid(y, x)
    
    ######## Konstanten ########
    Ωe = 7.2921e-5 # Drehfrequenz der Erde in 1/s
    Re = 6371e3 # Erdradius in m
    y0 = 3e6 # Breitengrad in Grad
    
    ########### Coriolisfaktor ##############
    latitude = 35
    θ_0 = np.deg2rad(latitude) # Breitengrad in Bogenmaß
    fc_0 = 2*Ωe*np.sin(θ_0) # Mittlere Zentrifugalkraft in 1/s^2
    f = fc_0 + (2*Ωe/Re)*(Y-y0) # Coriolisfaktor
    
    
    if darstellung==1:
        #Barotropische Instabilität
        W = 10000 - 500 * np.tanh(3e-3 * (Y - y0))
        W += np.random.uniform(1, 5, size=W.shape);
        B = np.zeros((Nx, Ny), dtype=np.double)
        
    elif darstellung==2:   
        #Rossby Wellen in der nördlichen Hemisphäre
        westwind = 30
        W = 10000 + (westwind/g)*f*(Y-y0)
        
        sigma_x = 5*dx
        sigma_y = 5*dy

        B = 4000*np.exp(-0.5*((X-10000000)/sigma_x)**2-0.5*((Y-y0)/sigma_y)**2) #https://en.wikipedia.org/wiki/Gaussian_function
    
    
    [dWdx, dWdy] = np.gradient(W, *[dy, dx])
    u = (-g/f)*dWdy
    v = (g/f)*dWdx
    # Initialisierung der Arrays
    h = W-B
    hu = h*u
    hv = h*v 

    return h, hu, hv, f

def erhaltungsschema_2D(h, hu, hv, CFL, Nx, Ny, darstellung):
    x = np.linspace(0, 10, Nx)
    y = np.linspace(0, 10, Ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Berechnung der Zeit
    z = 0 
    tmax = 5

    
    # Matrizen Berechnen der F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c
    
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
   

    if darstellung == 3:
        fig = plt.figure(figsize=(10,10))
    if darstellung == 2:
        fig = plt.figure(figsize=(20,10))
        ax_contour = fig.add_subplot(111,frameon = False)
        plt.show(block= False)

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

        # Berechnung der F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c
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

        #Reflektierende Randbedingungen
        h = reflektierender_block(h)
        hu = reflektierender_block(hu)
        hv = reflektierender_block(hv)
        u = hu/h
        v = hv/h
        v1 = np.append(v1, [np.amax(h)])
        t1 = np.append(t1, [z])
     
        # #create a meshgrid
        X,Y = np.meshgrid(x,y)

        if darstellung == 3:
            #plot the surface in 3D
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

def maccormack(h, hu, hv, f, CFL, Nx, Ny, darstellung, aufgabe):
    if (aufgabe == 3.2):
        x = np.linspace(0, 10, Nx)
        y = np.linspace(0, 10, Ny)
    elif(aufgabe == 3.3):
        interval = 100e3
        x = np.arange(0, interval * Nx, interval)
        y = np.arange(0, interval * Ny, interval)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Berechnung der Zeit
    z = 0
    if aufgabe == 3.2:
        tmax = 5
    if aufgabe == 3.3:
        tmax = 5 * 3600


    # Matrizen Berechnen der F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c

    Fa = np.zeros((Nx, Ny), dtype = np.double)
    Fb = np.zeros((Nx, Ny), dtype = np.double)
    Fc = np.zeros((Nx, Ny), dtype = np.double)
    Ga = np.zeros((Nx, Ny), dtype = np.double)
    Gb = np.zeros((Nx, Ny), dtype = np.double)
    Gc = np.zeros((Nx, Ny), dtype = np.double)
    h_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    hu_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    hv_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fa_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fb_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Fc_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Ga_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Gb_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    Gc_12 = np.zeros((Nx-1, Ny-1), dtype = np.double)
    v2 = np.zeros(1)
    t2 = np.zeros(1)
    v2[0] =  np.amax(h)

    if darstellung == 3:
        fig = plt.figure(figsize=(10,10))
    if darstellung == 2:
        fig = plt.figure(figsize=(20,10))
        ax_contour = fig.add_subplot(111, frameon=False)
        plt.show(block= False)
    
    while z < tmax:
        # Zeitschritt laut Jojo:
        print(h)
        # könnte entfernt werden zu np.array([0,0]), da jetzt auch in schleife 0,0 berechnet
        EWX = np.array([hu[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hu[0,0]/h[0,0]+np.sqrt(g*h[0,0])]) # Quelle: S.34 (3.5)
        EWY = np.array([hv[0,0]/h[0,0]-np.sqrt(g*h[0,0]), hv[0,0]/h[0,0]+np.sqrt(g*h[0,0])])
        for j in range(0,Nx):
            for k in range(0,Ny):
                EWX = np.append(EWX,[hu[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hu[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
                EWY = np.append(EWY,[hv[j,k]/h[j,k]-np.sqrt(g*h[j,k]), hv[j,k]/h[j,k]+np.sqrt(g*h[j,k])])
        dt = CFL * min(dx,dy)/(max(np.amax(EWX), np.amax(EWY))) # Quelle: S. 13 (1.58)
        print(dt)

        z += dt

        if aufgabe == 3.2:
            S_b = -f * hv
            S_c = f * hu
        elif aufgabe == 3.3:
            # S_b = -g*h*dBdx - f * hv
            # S_c = -g*h*dbdx - f * hu
            
            S_b = - f * hv
            S_c = - f * hu
        
        # Berechnung der F_j12a, F_j12b, F_j12c und  G_k12a, G_k12b, G_k12c
        for j in range(0, Nx):
            for k in range(0, Ny):
                Fa[j,k] = hu[j,k] # Quelle: S.4 (1.1)
                Fb[j,k] = (hu[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)
                Fc[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                Ga[j,k] = hv[j,k] # Quelle: S.4 (1.1)
                Gb[j,k] = (hu[j,k]*hv[j,k])/(h[j,k])
                Gc[j,k] = (hv[j,k]**2)/(h[j,k]) + 0.5*g*(h[j,k]**2)

        for j in range(0, Nx-1):
            for k in range(0, Ny-1):
                h_12[j,k]  = h[j,k]   - (dt/dx) * (Fa[j+1,k] - Fa[j,k]) - ((dt/dy) * (Ga[j,k+1] - Ga[j,k])) # Quelle: TUT
                hu_12[j,k] = hu[j,k]  - (dt/dx) * (Fb[j+1,k] - Fb[j,k]) - ((dt/dy) * (Gb[j,k+1] - Gb[j,k])) + (dt * S_b[j, k])
                hv_12[j,k] = hv[j,k]  - (dt/dx) * (Fc[j+1,k] - Fc[j,k]) - ((dt/dy) * (Gc[j,k+1] - Gc[j,k])) + (dt * S_c[j, k])

        for j in range(0, Nx-1):
            for k in range(0, Ny-1):
                Fa_12[j,k] = hu_12[j,k] # Quelle: S.4 (1.1)
                Fb_12[j,k] = (hu_12[j,k]**2)/(h_12[j,k]) + 0.5*g*(h_12[j,k]**2)
                Fc_12[j,k] = (hu_12[j,k]*hv_12[j,k])/(h_12[j,k])
                Ga_12[j,k] = hv_12[j,k] # Quelle: S.4 (1.1)
                Gb_12[j,k] = (hu_12[j,k]*hv_12[j,k])/(h_12[j,k])
                Gc_12[j,k] = (hv_12[j,k]**2)/(h_12[j,k]) + 0.5*g*(h_12[j,k]**2)

        # Berechnung der h, hu und hv
        for j in range(1, Nx-1):
            for k in range(1, Ny-1):
                h[j,k]   = 0.5 * (h[j,k]  +  h_12[j,k])  - (0.5 * (dt/dx) * (Fa_12[j,k] - Fa_12[j-1,k])) - (0.5*(dt/dy) * (Ga_12[j,k] - Ga_12[j,k-1])) # Quelle: TUT
                hu[j,k]  = 0.5 * (hu[j,k] + hu_12[j,k])  - (0.5 * (dt/dx) * (Fb_12[j,k] - Fb_12[j-1,k])) - (0.5*(dt/dy) * (Gb_12[j,k] - Gb_12[j,k-1])) + (dt*0.5*S_b[j, k])
                hv[j,k]  = 0.5 * (hv[j,k] + hv_12[j,k])  - (0.5 * (dt/dx) * (Fc_12[j,k] - Fc_12[j-1,k])) - (0.5*(dt/dy) * (Gc_12[j,k] - Gc_12[j,k-1])) + (dt*0.5*S_c[j, k])

        if aufgabe == 3.2:
        #Reflektierende Randbedingungen
            h = reflektierender_block(h)
            hu = reflektierender_block(hu)
            hv = reflektierender_block(hv)
        if aufgabe == 3.3:
            h = reflektierender_block(h)
            hu = reflektierender_block(hu)
            hv = reflektierender_block(hv)

        u = hu/h
        v = hv/h
        v2 = np.append(v2, [np.amax(h)])
        t2 = np.append(v2, [z])

        # #create a meshgrid
        if (aufgabe==3.2):
            X,Y = np.meshgrid(x,y)
        elif (aufgabe==3.3):
            Y,X = np.meshgrid(y,x)

        if darstellung == 3:
            #plot the surface in 3D
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
        if darstellung == 2:
            ax_contour.cla()
            ax_contour.set_title('Höhenverlauf')
            ax_contour.set_xlim(0, 240e4)
            ax_contour.set_ylim(0, 600e4)

            contour = ax_contour.contourf(X, Y, h, vmin=9.5e3, vmax=10.5e3 , shading='auto', cmap='jet')
            cb = fig.colorbar(contour, ax=ax_contour)
            # ax_cotour.pcolormesh(X, Y, h, shading='auto', vmax=2, vmin=1.5, cmap ='jet')
            ax_contour.quiver(X, Y, u, v)

            plt.draw()
            plt.pause(0.01)
            cb.remove()
    return h, hu, hv, v2, t2

# Irgendwo müssen noch Fehler in den Gleichungen sein, die Berechnung liefert negative Höhen, was nicht geht. Auch Höhen von größer als 2 sind am Anfang, das kann ja am Anfang auch nicht sein.

if __name__ == "__main__":
    # # Lax-Friedrich
    # h, hu, hv = anfangsbedingungen32(hh=2, ht=1.5, Nx=50, Ny=50)
    # h, hu, hv, v1, t1 = erhaltungsschema_2D(h, hu, hv, CFL=0.4, Nx = 50, Ny = 50, darstellung=3)
    # #Maccormack
    # h, hu, hv = anfangsbedingungen32(hh = 2, ht = 1.5, Nx = 50, Ny = 50)
    # f = np.zeros([50, 50])
    # h, hu, hv, v2, t2 = maccormack(h, hu, hv, f, CFL=0.4, Nx = 50, Ny = 50,  darstellung= 3, aufgabe= 3.2)
    # Aufgabenteil 3.3.1
    h, hu, hv, f = anfangsbedingungen33()
    h, hu, hv, v3, t3 = maccormack(h, hu, hv, f, CFL=0.4, Nx = 24, Ny = 60,  darstellung= 2, aufgabe= 3.3)

    plt.style.use('seaborn')

    # # Vergleich der Lösungen
    # plt.plot(t2, v2, label='MacCormack')
    # plt.plot(t1, v1, label='Lax-Friedrich')
    # plt.title('MacCormack vs Lax-Friedrich')
    # plt.xlabel('Zeit (s)')
    # plt.ylabel('Höhe (m)')
    # plt.legend()
    # plt.show()