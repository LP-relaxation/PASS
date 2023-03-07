###############################################################################

# Bisection Parallel Adaptive Survivor Selection (Synchronized)
#	a.k.a. bi-PASS Sync

# Linda Pei 2023

###############################################################################

###########################
######## IMPORTS ##########
###########################

import time
import sys
import numpy as np
from mpi4py import MPI
import pandas as pd

# sys.argv[0] is the name of the script
output_file_name = str(sys.argv[1])
config = __import__(sys.argv[2])
trial = int(sys.argv[3])

#########################################
######## biPASS_sync Algorithm ##########
#########################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

def create_slicepoints(num_working_workers, num_contenders):
	# (leftover) workers are assigned (base_assignment + 1) number of systems
	# (num_working_workers - leftover) workers get (base_assignment) systems
	leftover = num_contenders % num_working_workers
	base_assignment = int(np.floor(num_contenders/float(num_working_workers)))

	# slicepoints is a tuple of length (num_working_workers + 1)
	# slicepoints are used to "slice" the set of contenders
	# worker i is assigned to simulate contenders[slicepoints[worker_id]]
	#	through contenders[slicepoints[worker_id]+1] (consecutively)
	slicepoints = np.append([0], 
		np.cumsum(np.append(np.full(leftover, base_assignment+1), 
			np.full(num_working_workers - leftover, base_assignment))))
	return tuple(slicepoints)

def biPASS(trial):

	trial_start_time = time.clock()

	status = MPI.Status()

	base_bit_generator = np.random.MT19937(0).jumped(int((trial + 1) * num_workers))
	for worker_id in range(num_workers):	
		if rank == worker_id:
			worker_bit_generator = base_bit_generator
			base_bit_generator = base_bit_generator.jumped()
	
	good_systems = np.where(np.array(config.true_means) >= config.true_standard)[0].tolist()
	num_systems = len(config.true_means)
	
	##########################
	######## MASTER ##########
	##########################

	if rank == master_rank:
		
		################################
		######## MASTER SETUP ##########
		################################

		if config.known_variance == True:
			est_variances = np.array(config.true_variances)
			scaling_factors = est_variances
		else:
			est_variances = np.full(num_systems, np.inf)
			scaling_factors = np.full(num_systems, np.inf)
		
		running_sums = np.full(num_systems, 0.0)
		reps = np.full(num_systems, 0.0)
		run_lengths = np.full(num_systems, 0.0)
		
		standard = config.init_standard
	
		contenders = np.arange(num_systems)
		good_contenders = np.array([system for system in good_systems])

		num_working_workers = num_workers

		#############################
		######## MAIN LOOP ##########
		#############################

		cycle = 0

		while cycle < config.num_cycles and len(contenders) > 1:

			cycle += 1

			run_lengths[contenders] = np.full(len(contenders), 
				config.run_length(reps[contenders][0]))

			######################
			######## Send ########
			######################

			# If more working workers than contenders, retire excess workers
			if num_working_workers > len(contenders):
				for worker_id in range(len(contenders), num_working_workers):
					comm.send(np.array([None]), dest = worker_id)	
				num_working_workers = len(contenders)

			slicepoints = create_slicepoints(num_working_workers, len(contenders))

			for worker_id in range(num_working_workers):
				worker_assignment_indices = contenders[slicepoints[worker_id]:slicepoints[worker_id+1]]
				data_to_send = np.append(worker_assignment_indices, run_lengths[worker_assignment_indices])
				comm.send(data_to_send, dest = worker_id)

			num_pending_messages = num_working_workers

			while num_pending_messages > 0:
				
				data = comm.recv(source = MPI.ANY_SOURCE, status = status)
				worker_id = status.Get_source()

				worker_assignment_indices = contenders[slicepoints[worker_id]:slicepoints[worker_id+1]]

				running_sums[worker_assignment_indices] += data[:slicepoints[worker_id+1] - slicepoints[worker_id]]
				reps[worker_assignment_indices] += run_lengths[worker_assignment_indices]

				# n0 round: estimate variances
				if cycle == 1 and config.known_variance == False:
					est_variances[worker_assignment_indices] = data[slicepoints[worker_id+1]-slicepoints[worker_id]:]

				num_pending_messages -= 1

			##############################
			## Post-simulation Updating ##
			##############################

			if cycle  == 1:
				for job_id in range(num_systems):
					if config.scaling_type == "custom":
						scaling_factors[job_id] = est_variances[job_id]
					elif config.scaling_type == "pooled":
						scaling_factors[job_id] = np.average(est_variances)
				if config.known_variance == False:
					np.savetxt("est_variances_" + str(trial) + str(output_file_name) + ".csv", est_variances, delimiter=",")

			if config.init_standard == -np.inf:
				standard = config.update_standard(running_sums, reps, contenders)

			contenders = contenders[(np.divide(running_sums[contenders] - reps[contenders] * standard, 
				scaling_factors[contenders]) > config.boundary_function(np.divide(reps[contenders], scaling_factors[contenders])))]
			good_contenders = np.intersect1d(contenders, good_systems)

		for worker_id in range(num_working_workers):
			comm.send(np.array([None]), dest = worker_id)
		new_summary_row = pd.DataFrame([[len(contenders), len([system for system in good_systems if system not in contenders]), cycle, np.sum(reps), time.clock() - trial_start_time]], 
			columns=["Contenders", "False Eliminations", "Cycles", "Total Effort", "Time"])
		new_summary_row.to_csv("biPASS_sync_master_summary_" + str(output_file_name) + ".csv", mode="a", header=(trial==0))
		
		return 0
		
	##########################
	######## WORKER ##########
	##########################
	
	else: # if worker

		cycle = 0

		while cycle <= config.num_cycles:

			cycle += 1

			data_received = comm.recv(source = master_rank)
			num_jobs = int(np.floor(len(data_received)/2))

			if num_jobs > 0:
				simulation_jobs = data_received[:num_jobs]
				run_lengths = data_received[num_jobs:]
				sums = np.zeros(num_jobs)

				if cycle == 1 and config.known_variance == False:
					est_variances = np.zeros(num_jobs)
				
				for i in range(num_jobs):
					z = 0
					output_history = []
					while z < run_lengths[i]:
						output = config.simulation_model(simulation_jobs[i], worker_bit_generator)
						output_history.append(output)
						z += 1
					sums[i] += np.sum(output_history)
					if cycle == 1 and config.known_variance == False:
						est_variances[i] = np.var(output_history)

				if cycle == 1 and config.known_variance == False:
					data_to_send = np.append(sums, est_variances)
				else:
					data_to_send = sums
				
				comm.send(data_to_send, dest = master_rank)

			elif num_jobs == 0:
				return 0 

#############################
######## Execution ##########
#############################

if rank == master_rank:
	print("Running trial " + str(trial))
biPASS(trial)
