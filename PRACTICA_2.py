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

plt.close('all')

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
    sds = []
    for i in range(4):
    
        # PID 1
        if i == 0:
            ts = 3.6
            tp = 7
        
        if i == 1:
        # PID 2
            ts = 3.6
            tp = 1
      
        if i == 2:
        # PID 3
            ts = 1.9
            tp = 1
        
        if i == 3:
        # PID 4
            ts = 3.3
            tp = 0.5
        
        C_5 = 3     # 5% criterion
    
        # Design point (complex plane)
        sigma = C_5 / ts
        w_d = np.pi / tp
            
        # tan_theta = -np.pi/(np.log(Mp))
        # w_d = sigma*tan_theta
        
        sds.append(complex(-sigma,w_d))
        
    print(f'Sd={sds}')
    
    ts = 3.6
    tp = 1
    # Design point (complex plane)
    sigma = C_5 / ts
    w_d = np.pi / tp

    sd = (complex(-sigma,w_d)) 
   
#%% Linearized system using small angles approximation
    
    # We stabilize the system around theta = 0.
    
    # We will assume that the system stays within a small neighborhood of this equillbrium in theta = pi
    # Where: sin(theta) = -theta, cos(theta) ~ -1 and theta_dot² ~ 0
    
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
   
    plt.plot(-0.83333, 0.4487, marker='x', markersize=10, label='PID 1')
    plt.plot(-0.83333, 0.415, marker='x', markersize=10, label='PID 2')
    plt.plot(-0.5789, 3.14159, marker='x', markersize=10, label='PID 3')
    plt.plot(-0.9090, 6.28318, marker='x', markersize=10, label='PID 4')
    control.root_locus(G)
    plt.legend()
    
    # plt.plot(-sigma, w_d, marker='x', markersize=10, markerfacecolor='g')
    
    # Original system step response
    plt.figure()
    t = np.linspace(0,1,100)
    _, theta = control.step_response(G, T=t)
    plt.plot(t,theta)
    plt.title('$\\theta$ variation along time in Linealized system')
    plt.grid(alpha=0.3)
    plt.ylabel('$\\theta$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
  
    
    #%% PID controller
    
    pids = []
    plt.figure()
    for sd in sds:
            
        #Only valid for this G transfer function
        numerator = control.tf([4.54545455, 0],[1])
        denominator = control.tf([1, 0.18181818,-31.21363636,-4.45909091],[1])
            
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
        
        pids.append(K*Gc)
        
        t, theta = control.step_response(T)
        plt.plot(t,theta, label=f'PID {sds.index(sd)+1}')
    
    plt.title('$\\theta$ variation along time in Linealized system with PID controller')
    plt.grid(alpha=0.3)
    plt.ylabel('$\\theta$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()
    
    #%% Perturbations with PID controller
    
    # Perturbation response 
    P = np.random.uniform(-0.25,0.25)
    print(f'Perturbation force is: {P} N')
        
    plt.figure() 
    for pid in pids:
        
        H = 1
        theta_ref = 0
        
        aux1 = control.parallel(theta_ref, -H)
        system = control.feedback(G,pid*aux1,sign=1)
        
        t, out = control.step_response(system)
        plt.plot(t,P*out, label=f'PID {pids.index(pid)+1}')
        
    plt.title(f'Perturbation response with PID controller and p(t)={np.round(P,3)}u(t) N')
    plt.grid(alpha=0.3)
    plt.ylabel('$\\theta$ (rad)')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()

    
    #%% Calculating IAE, ISE, ITSE, ITAE
    
    time = np.linspace(0,8,1000)
    errors = []
    plt.figure()
    for pid in pids:
        T = control.feedback(pid*G, H);
        _, theta_out = control.step_response(T, T=time)
    
        error = 1-theta_out
        errors.append(error)
        plt.plot(time, abs(error), label=f'absolute error PID={pids.index(pid)+1}')
        
    
    plt.title(f'Absolute error variation along time')
    plt.grid(alpha=0.3)
    plt.ylabel('error')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()
    
    # ISE
    plt.figure()

    for pid in pids:
        count = 0
        ISE = []
        
        for i in errors[pids.index(pid)]**2:
            count = count + i 
            ISE.append(count)
        
        plt.plot(time, ISE, label='ISE')
    
    plt.title('ISE')
    plt.grid(alpha=0.3)
    plt.ylabel('ISE')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()
    
    # IAE
    plt.figure()

    for pid in pids:
        count = 0
        IAE = []
        
        for i in np.abs(errors[pids.index(pid)]):
            count = count + i 
            IAE.append(count)
        
        plt.plot(time, IAE, label='IAE')
    
    plt.title('IAE')
    plt.grid(alpha=0.3)
    plt.ylabel('IAE')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()
    
     # ITSE
    plt.figure()

    for pid in pids:
        count = 0
        ITSE = []
        
        for i in time*errors[pids.index(pid)]**2:
            count = count + i 
            ITSE.append(count)
        
        plt.plot(time, ITSE, label='ITSE')
    
    plt.title('ITSE')
    plt.grid(alpha=0.3)
    plt.ylabel('ITSE')
    plt.xlabel('time (s)')
    plt.show()
    plt.legend()
    
    # ITAE
    plt.figure()

    for pid in pids:
        count = 0
        ITAE = []
        
        for i in time*np.abs(errors[pids.index(pid)]):
            count = count + i 
            ITAE.append(count)
        
        plt.plot(time, ITAE, label='ITAE')
    
    plt.title('ITAE')
    plt.grid(alpha=0.3)
    plt.ylabel('ITAE')
    plt.xlabel('time (s)')
    plt.show()
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