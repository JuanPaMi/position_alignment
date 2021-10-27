import numpy as np
import sys
import matplotlib.pyplot as plt

file = sys.argv[1] 
timestep = 10000
N_particles = int(sys.argv[2])
N_obstacles = int(sys.argv[3])
noise = input('enter noise value: ')
N_tot = N_obstacles + N_particles
# particles  ptype = 0 ; obstacles ptype= 1

num_lines = int(sum(1 for _ in open(file)))

num_confs = num_lines // (N_tot + 2)

# Unwrap trajectories

box_x = box_y = 81



X,Y = np.empty(0),np.empty(0)
for n in range(num_confs):
    print(n)
    particle_type,x,y,vx,vy = np.loadtxt(file,skiprows=n*N_tot + 2*(n+1),
    unpack= True,max_rows=N_tot)
    particles_mask = particle_type==0
    obstacles_mask = particle_type==1
    x_aux = x[particles_mask]
    y_aux = y[particles_mask]
    X = np.append(X,x_aux)
    Y = np.append(Y,y_aux)


print(len(X),len(Y),num_confs*N_particles)

position_x = np.zeros((N_particles,num_confs))
position_y = np.zeros((N_particles,num_confs))

for i in range(0,N_particles):
    print(i)
    x_unwrapped = X[i]
    y_unwrapped = Y[i]
    for t in range(1,num_confs):
        tid = t*N_particles + i 
        x_unwrapped = X[tid] + np.rint((x_unwrapped - X[tid])/box_x)*box_x
        y_unwrapped = Y[tid] + np.rint((y_unwrapped - Y[tid])/box_y)*box_y
        X[tid] = x_unwrapped
        Y[tid] = y_unwrapped
        position_x[i,t] = X[tid]
        position_y[i,t] = Y[tid]

msd = np.zeros(num_confs//2)
tau = np.zeros(num_confs//2)
for m in range(1,len(msd)):
    count = 0
    sum = 0
    for k in range(len(msd)-m):
        sum += (position_x[:,m+k] - position_x[:,m])**2 + (position_y[:,m+k] - position_y[:,m])**2 
        count += 1
    
    msd[m] = sum.mean() / count
    tau[m] = m



np.savetxt('data_msd'+str(N_particles)+'particles_'+str(N_obstacles)+'obstacles'+noise,
(tau,msd),header='time\tmsd')

plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.plot(tau,msd)
plt.xlabel('timesteps')
plt.ylabel('msd')
plt.savefig('msd.png')
plt.show()
