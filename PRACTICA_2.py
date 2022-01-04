#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:17:42 2021

@author: alumno
"""

# Inverted pendulum analysis

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.integrate import odeint
import control

#%% Parameters

if __name__ == "__main__":
    plt.close('all')
    
    # System Parameters (parametrized). 
    
    u = 1      # Similar to a step reponse
    g = 9.81    
    L = 0.3 
    m = 0.2 # 0.25
    I = 0.006
    M = 0.5 # 2.4
    b = 0.1
    
    R = np.sqrt(2*I/m)
      
    # Time vector
    t = np.linspace(0,1,50000)
    
    # Initial conditions
    xRef = np.array([0.0,     # x_0 
                     0.0,     # dot_x_0
                     np.pi,   # theta_0
                     0.0])    # dot_theta_0
    
    deltaX   = np.array([0.0,   # x_init 
                         0.0,   # dot_x_init
                         0.0,   # theta_init
                         0.0])  # dot_theta_init
  

    # Establishment time: ts
    
    # Calculating t* = t_lim
    t_lim = (((M+m)/(b*g))*np.sqrt((I+m*L**2)/(m)))**(1/3)
    
    ts_min = 3*t_lim
    ts_max = 6*t_lim
    print(f'\n{ts_min} <= ts <= {ts_max}')
    
    # Peak time: tp 
    tp_max = 10*t_lim
    print(f'tp <= {tp_max}')
    
    # Selecting design values
    
    # PID 1
    # ts = 3.6
    # tp = 7
    
    # PID 2
    ts = 3.6
    tp = 1
    
    # PID 3
    # ts = 1.9
    # tp = 1
    
    C_5 = 3     # 5% criterion

    # Design point (complex plane)
    sigma = C_5 / ts
    w_d = np.pi / tp
        
    # tan_theta = -np.pi/(np.log(Mp))
    # w_d = sigma*tan_theta
    
    sd = complex(-sigma,w_d)
    
    print(f'Sd={sd}')

#%% Linearized system using small angles approximation
    
    # We stabilize the system around theta = pi -> tetha = pi + phi, where phi is de deviation
    # from the equilibrium point theta = pi.
    
    # We will assume that the system stays within a small neighborhood of this equillbrium in theta = pi
    # Where: sin(theta) = sin(pi + phi)~ -phi, cos(theta) = cos(pi + phi) ~ -1 and theta_dot² = pi² * phi_dot² ~ 0
    
    def linealizeSystem(params):
        
        # System constants
        M,m,L,b,u,R,g = params

        # Where dot_x2 = -b*gamma*x_2 + beta*x_3+gamma*u
        gamma = (I+m*L**2)/(I*(M+m)+M*m*L**2)
        beta = (m**2*L**2*g)/(I*(M+m)+M*m*L**2)
        
        # Where dot_x4 = -b*phi*x_2 + alpha*x_3+phi*u
        tau = (m*L)/(I*(M+m)+M*m*L**2)
        alpha = (m*g*L*(m+M))/(I*(M+m)+M*m*L**2)
    
        # Creating Space state
        A = np.array([[0,1,0,0],[0,-b*gamma,beta,0],[0,0,0,1],[0,-b*tau,alpha,0]])
        B = np.array([[0],[gamma],[0],[tau]])
        C = np.array([0,0,1,0])
        D = np.array([[0]])
    
        return A,B,C,D
    
    A,B,C,D = linealizeSystem([M,m,L,b,u,R,g])
    linearizedSystem = control.ss(A,B,C,D)
    
    # Original System
    G = control.ss2tf(linearizedSystem)
    
    # Removing values near to 0
    den = np.around(G.den,decimals=12)
    num = np.around(G.num,decimals=12)
    G= control.tf(num,den)
    G_poles = G.pole()
    G_zeros = G.zero()
    
    print(f'\nLinealized system transfer function is:\n{G}')
    print(f'\nLinealized system Poles at:\n {G_poles}')
    print(f'\nLinealized system Zeros at:\n {G_zeros}')


    # Original system root locus
    plt.figure()
    control.root_locus(G)
    plt.plot(-sigma,w_d, marker='x', markersize=10, markerfacecolor='g')
    
    # Original system step response
    plt.figure()
    t, theta = control.step_response(G)
    plt.plot(t,theta)
  
    
    #%% PID controller
    
    ################ only valid for this G transfer function
    numerator = control.tf([4.54545455, 0],[1])
    denominator = control.tf([1, 0.18181818,-31.21363636,-4.45909091],[1])
    ##############
    
    argumento = np.angle(numerator(sd), deg=True) - np.angle(denominator(sd), deg=True)
    print(f'argumento: {argumento} degrees')
    
    # We add a pole in zero and calculate beta.
    beta_prima = 180 + np.angle(denominator(sd)*sd, deg=True) - np.angle(numerator(sd), deg=True)
    print(f'beta: {beta_prima} degrees')
    
    # We establish two poles to satisfy the argument criterion
    a = w_d/(np.tan(np.deg2rad(55))) + sigma
    b = w_d/(np.tan(np.deg2rad(beta_prima-55))) + sigma
    
    print(f'\nAdd pole in a: {a}')
    print(f'Add pole in b: {b}')
    
    num = np.flip(np.polynomial.Polynomial.fromroots((-a,-b)).coef)  
    #Gc = control.tf([1,7,12],[1,0])
    Gc = control.tf(num,[1,0])
    
    print(f'\nController, Gc is = {Gc}')
    
    arg = np.round(np.angle(G(sd)*Gc(sd), deg=True),4)
    print(f'Argument criterion with Gc: {arg} degrees')
    
    # Now, we adjust gain K
    
    # K calculation
    T_s = G*Gc
    T_sd = T_s(sd)
    K = 1/np.abs(T_sd)
    
    print(f'K is: {K}')
    T = control.feedback(G,K*Gc);
    
    #%% Perturbations with PID controller
    
    H = 1
    theta_ref = 0
    
    aux1 = control.parallel(theta_ref, -H)
    aux2 = control.series(aux1, Gc)
    system = control.feedback(G,K*aux2,sign=1)
    
    # Perturbation response 
    P = np.random.uniform(-0.25,0.25)
    print(f'Perturbation force is: {P} N')
    
    plt.figure()    
    t, out = control.step_response(system)
    plt.plot(t,P*out)
    plt.title('Perturbation response with PID controller')
    plt.xlabel('time (s)')
    plt.ylabel('$\\theta$ (rad)')
    
    #%% Calculating IAE, ISE, ITSE, ITAE
    
    T_prima = control.feedback(K*Gc*G, H);
    time, theta_out = control.step_response(T_prima)
    
    # theta_ref = 0 -> inverted pendulum
    plt.figure()
    error = 1-theta_out
    plt.plot(time, error, label='error')
    plt.plot(time, np.abs(error), label='absolute error')
    plt.legend()
    
    
    plt.figure()
    # ISE
    count = 0
    ISE = []
    for i in error**2:
        count = count + i 
        ISE.append(count)
    
    plt.plot(time, ISE, label='ISE')
    
    # IAE
    count = 0
    IAE = []
    for i in np.abs(error):
        count = count + i 
        IAE.append(count)
    
    plt.plot(time, IAE, label='IAE')
    
    # ITSE
    count = 0
    ITSE = []
    for i in time*error**2:
        count = count + i 
        ITSE.append(count)
    
    plt.plot(time, ITSE, label='ITSE')
    
    # ITAE
    count = 0
    ITAE = []
    for i in time*np.abs(error):
        count = count + i 
        ITAE.append(count)
    
    plt.plot(time, ITAE, label='ITAE')
    plt.legend()    
   

    
#%% Non-linear

# Non-linear function matrix
def f(xVec, t, params):
    
    # State Variables vector
    # x1 = x, x2 = dX, x3 = theta, x4 = dTheta
    x, dX, theta, dTheta = xVec
    
    # System parameters
    M,m,L,b,u,R,g = params
    
    # x_dot = f1 
    # x_dot_dot = f2
    # theta_dot = f3
    # theta_dot_dot = f4
    
    return [dX,
            (((-2*L**2-R**2)*(-b*dX+dTheta**2*L*m*np.sin(theta)+u))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M) - (2*g*L**2*m*np.cos(theta)*np.sin(theta))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M)),
            dTheta,
            ((2*L*(-dX*b+dTheta**2*L*m*np.sin(theta)+u)*np.cos(theta))/(2*L**2*m*np.cos(theta)**2-m*R**2-M*R**2-2*L**2*m-2*L**2*M) - (g*L*m*(-2*m-2*M)*np.sin(theta))/(2*L**2*m**2*np.cos(theta)**2-2*L**2*m**2-m**2*R**2-m*M*R**2-2*L**2*m*M))
            ]



    #%% Calculating Non linear sistem (EDOs)
      
    xNL = odeint(f, xRef + deltaX, t, args=([M,m,L,b,u,R,g],))

   #%%
   
    """# practica 2
    def linealizeSystem(params):
        
        # System constants
        M,m,L,b,u,R,g = params

        # Where dot_x2 = -b*gamma*x_2 + beta*x_3+gamma*u
        gamma = (I+m*L**2)/(I*(M+m)+M*m*L**2)
        beta = -(m**2*L**2*g)/(I*(M+m)+M*m*L**2)
        
        # Where dot_x4 = -b*phi*x_2 + alpha*x_3+phi*u
        tau = -(m*L)/(I*(M+m)+M*m*L**2)
        alpha = (m*g*L*(m+M))/(I*(M+m)+M*m*L**2)
    
        # Creating Space state
        A = np.array([[0,1,0,0],[0,-b*gamma,beta,0],[0,0,0,1],[0,-b*tau,alpha,0]])
        B = np.array([[0],[gamma],[0],[tau]])
        C = np.array([0,0,1,0])
        D = np.array([0])
    
        return A,B,C,D
    
    # practica 2 true
    def linealizeSystem(params):
        
        # System constants
        M,m,L,b,u,R,g = params

        # Where dot_x2 = -b*gamma*x_2 + beta*x_3+gamma*u
        v1 = (M+m)/(I*(M+m)+m*M*L**2)
        v2 = (I+m*L**2)/(I*(M+m)+m*M*L**2)
    
        # Creating Space state
        A = np.array([[0,1,0,0],[0,-b*v2,-((m**2*L**2*g*v2)/(I+m*L**2)),0],[0,0,0,1],[0,(m*L*b*v2)/(M+m),m*g*L*v1,0]])
        B = np.array([[0],[v2],[0],[-((m*L*v1)/(M+m))]])
        C = np.array([0,0,1,0])
        D = np.array([0])
    
        return A,B,C,D
    
       
    A,B,C,D = linealizeSystem([M,m,L,b,u,R,g])
    linearizedSystem = control.ss(A,B,C,D)
    
    # Original System
    G = control.ss2tf(linearizedSystem)
    
    # Removing values near to 0
    den = np.around(G.den,decimals=12)
    num = np.around(G.num,decimals=12)
    G= control.tf(num,den)
    G_poles = G.pole()
    G_zeros = G.zero()
    
    print(f'\nLinealized system transfer function is:\n{G}')
    print(f'\nLinealized system Poles at:\n {G_poles}')
    print(f'\nLinealized system Zeros at:\n {G_zeros}')"""

    """WEB:
    plt.figure(1)
    control.root_locus(G)
    
    add_pole = control.tf([1],[1,0])
    G_zero = G*add_pole
    plt.figure(2)
    control.root_locus(G_zero)
    
    # zeros in -3 and -4, pole in 0
    Gc = control.tf([1,7,12],[1,0])
    
    K = 20
    T = control.feedback(G,K*Gc)
    plt.figure(3)
    control.sisotool(T)
    
    print(Gc)
    print(G)
    print(T)"""
    
   
      
    """
    #%% PID controller design

    # Argument criterion
    
    arg = np.angle(G(sd), deg=True)
    print(f'\nargument {arg}º = 180º?')
    
    # PID controller: Gc = (s+a)(s+b)/s
    
    # b* calculation
    # b = np.abs(0.1 * (-0.14283166))
    Gc = control.tf([1,b],[1,0])

    
    
    #%%
    plt.figure()
    #control.sisotool(G*Gc)
    control.sisotool(T)
     
    plt.figure()
    control.root_locus(G*Gc)
    plt.plot(-sigma,w_d, marker='x', markersize=10, markerfacecolor='g')
    
    # Gc = control.tf([1,a],[1])
    
    plt.figure()
    t, out = control.step_response(T)
    plt.plot(t,out)

    ###############
    #%%
    
    # a calculation
        # Beta calculation:
  
    beta = 180 - np.angle(G(sd), deg=True)
    # a = np.abs(w_d/(np.tan(np.deg2rad(beta))) + sigma)
    
    # K calculation
    # T_s = G*Gc
    # T_sd = T_s(sd)
    #K = 1/np.abs(T_sd)
    K=1
    
  #  num = np.polynomial.Polynomial.fromroots((a, b)).coef
    num = np.flip(np.polynomial.Polynomial.fromroots((-10,-5)).coef)
    #K = 20
    
    #Gc = control.tf([1,7,12],[1,0])
     
    Gc = control.tf(num,[1,0])
  
    print(f'Gc is = {Gc}')
    
    T = control.feedback(G,K*Gc);
    
    plt.figure()
    #control.sisotool(G*Gc)
    control.sisotool(T)
     
    plt.figure()
    control.root_locus(T)
    plt.plot(-sigma,w_d, marker='x', markersize=10, markerfacecolor='g')
    
    # Gc = control.tf([1,a],[1])
  
    
    plt.figure()
    t, out = control.step_response(T)
    plt.plot(t,out)

    """