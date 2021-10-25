from matplotlib import use
import numpy as np
import sys
import matplotlib.pyplot as plt

file = sys.argv[1] 
timestep = 10000
N_particles = int(file.split('_')[0].replace('particles',''))
N_obstacles = int(file.split('_')[1].replace('obstacles',''))
noise = file.split('_')[2].replace('.xyz','')
N_tot = N_obstacles + N_particles
# particles  ptype = 0 ; obstacles ptype= 1

num_lines = int(sum(1 for _ in open(file)))

num_confs = num_lines // (N_tot + 2)

polar_order_parameter = np.zeros(num_confs)
nematic_order_parameter = np.zeros(num_confs)

for n in range(num_confs):
    print(n)
    particle_type,x,y,vx,vy = np.loadtxt(file,skiprows=n*N_tot + 2*(n+1),
    unpack= True,max_rows=N_tot)
    particles_mask = particle_type==0
    obstacles_mask = particle_type==1
    angles = np.arctan2(vy[particles_mask],vx[particles_mask])
    polar_order_parameter[n] = np.abs(np.exp(1j*angles).mean())
    nematic_order_parameter[n] = np.abs(np.exp(2j*angles).mean())

steps = np.arange(timestep,(num_confs+1)*timestep,timestep,dtype='float64')


np.savetxt('data_orderparameter'+
str(N_particles)+'particles_'+str(N_obstacles)+'obstacles'+noise,
(steps,polar_order_parameter,nematic_order_parameter),
header='time\tpolar\tnematic')

plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.plot(steps,polar_order_parameter)
plt.xlabel('timesteps')
plt.ylabel('polar order parameter')
plt.savefig('polarorderparameter_'+noise+'noise.png')
plt.show()

plt.style.use('classic')
plt.figure(figsize=(10,8))
plt.plot(steps,nematic_order_parameter)
plt.xlabel('timesteps')
plt.ylabel('nematic order parameter')
plt.savefig('nematicorderparameter_'+noise+'noise.png')
plt.show()