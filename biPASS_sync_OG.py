# 2022

# I should make this object oriented I think...

# Old note to assume that there are less workers than systems to start out with, i.e. p < k
# But do we actually need this assumption?

# master's est_variances different than worker's est_variances

# be careful about python indexing
# remember range is [,)

###########################
######## IMPORTS ##########
###########################

import time
import sys
import numpy as np
import randomgen
from mpi4py import MPI
import pandas as pd

# sys.argv[0] is the name of the script
output_file_name = str(sys.argv[1])
config = __import__(sys.argv[2])
trial = int(sys.argv[3])

np.seterr(all='raise')

#########################################
######## biPASS_sync Algorithm ##########
#########################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

def biPASS(trial):

	trial_start_time = time.clock()

	status = MPI.Status()

	base_bit_generator = randomgen.MT19937(0).jumped(int((trial + 1) * num_workers))
	for worker_id in range(num_workers):
		if rank == worker_id:
			worker_bit_generator = base_bit_generator.jumped(int(rank))
	
	good_systems = np.where(np.array(config.true_means) >= config.true_standard)[0].tolist()
	num_systems = len(config.true_means)
	
	##########################
	######## MASTER ##########
	##########################

	if rank == master_rank:
		
		################################
		######## MASTER SETUP ##########
		################################

		if config.known_common_variance == True:
			est_variances = np.array(config.true_variances)
			scaling_factors = np.full(num_systems, config.true_variances[0])
		else:
			# if not known or common
			est_variances = np.full(num_systems, np.inf)
			scaling_factors = np.full(num_systems, np.inf)
		
		running_sums = np.full(num_systems, 0.0)
		reps = np.full(num_systems, 0.0)
		run_lengths = np.full(num_systems, config.run_length(reps[0]))

		max_total_reps = config.max_total_reps[trial]
		
		standard = config.init_standard
	
		contenders = np.arange(num_systems)
		good_contenders = np.array([system for system in good_systems])

		num_working_workers = num_workers

		# print("Bing bong!")

		#############################
		######## MAIN LOOP ##########
		#############################

		cycle = 0

		while cycle < config.num_cycles and len(contenders) > 1:

			# print("Cycle " + str(cycle))

			cycle += 1

			# assuming all contenders have the same run length right now
			run_lengths[contenders] = np.full(len(contenders), config.run_length(reps[contenders][0]))

			######################
			######## send ########
			######################

			if num_working_workers > len(contenders):
				for worker_id in range(len(contenders), num_working_workers):
					comm.send(np.array([None]), dest = worker_id)	
				num_working_workers = len(contenders)

			# print(num_working_workers)

			leftover = len(contenders) % num_working_workers
			base_assignment = int(np.floor(len(contenders)/float(num_working_workers)))
			slices = np.append([0], np.cumsum(np.append(np.full(leftover, base_assignment+1), np.full(num_working_workers - leftover, base_assignment))))

			# print(leftover)
			# print(base_assignment)
			# print(slices)

			for worker_id in range(num_working_workers):
				worker_assignment_indices = contenders[slices[worker_id]:slices[worker_id+1]]
				comm.send(np.append(worker_assignment_indices, run_lengths[worker_assignment_indices]), dest = worker_id)
				# print(np.append(worker_assignment_indices, run_lengths[worker_assignment_indices]))
				# print(worker_id)

			num_pending_messages = num_working_workers

			while num_pending_messages > 0:
				
				data = comm.recv(source = MPI.ANY_SOURCE, status = status)
				worker_id = status.Get_source()

				worker_assignment_indices = contenders[slices[worker_id]:slices[worker_id+1]]

				running_sums[worker_assignment_indices] += data[:slices[worker_id+1]-slices[worker_id]]

				# n0 round: estimate variances
				if cycle == 1:
					est_variances[worker_assignment_indices] = data[slices[worker_id+1]-slices[worker_id]:]

				reps[worker_assignment_indices] += run_lengths[worker_assignment_indices]

				num_pending_messages -= 1

			##############################
			## post-simulation updating ##
			##############################

			if cycle  == 1:

				for job_id in range(num_systems):
					if config.scaling_type == "custom":
						scaling_factors[job_id] = est_variances[job_id]
					elif config.scaling_type == "pooled":
						scaling_factors[job_id] = np.average(est_variances)

				if config.known_common_variance == False:
					if config.output_mode == "profile":
						np.savetxt("initial_sample_" + str(trial) + str(output_file_name) + ".csv", np.divide(running_sums, reps), delimiter=",")
						np.savetxt("scaling_factors_" + str(trial) + str(output_file_name) + ".csv", scaling_factors, delimiter=",")

			if config.init_standard == -np.inf:
				standard = config.update_standard(running_sums, reps, contenders)

			# print(standard)
			
			# check all contenders for elimination at once
			contenders = contenders[(np.divide(running_sums[contenders] - reps[contenders] * standard, scaling_factors[contenders]) > config.boundary_function(np.divide(reps[contenders], scaling_factors[contenders])))]
			good_contenders = np.intersect1d(contenders, good_systems)

			if cycle % 500 == 0:
				print((standard, len(contenders)))

			# if not all the cycles have retired but there is only 1 contender left
			if cycle == config.num_cycles or np.sum(reps) >= max_total_reps or len(contenders) == 1:
				# print("Terminating...")
				for worker_id in range(num_working_workers):
					comm.send(np.array([None]), dest = worker_id)
				new_summary_row = pd.DataFrame([[len(contenders), len([system for system in good_systems if system not in contenders]), cycle, np.sum(reps), time.clock() - trial_start_time]], columns=["Contenders", "False Eliminations", "Cycles", "Total Effort", "Time"])
				new_summary_row.to_csv("biPASS_sync_master_summary_" + str(output_file_name) + ".csv", mode="a", header=(trial==0))
				return 0
		
	##########################
	######## WORKER ##########
	##########################
	
	else: # if worker

		cycle = 0

		# can also do while cycle <= config.num_cycles:
		# it's a <= not a < because still need to receive the master termination message! there are different ways to do this
		while True:

			cycle += 1

			data = comm.recv(source = master_rank)
			num_jobs = int(np.floor(len(data)/2))

			if num_jobs > 0:

				simulation_jobs = data[:num_jobs]
				run_lengths = data[num_jobs:]

				sums = np.zeros(num_jobs)

				# n0 round: estimate variances
				if cycle == 1:
					est_variances = np.zeros(num_jobs)

					for i in range(num_jobs):
						z = 0
						output_history = []
						
						while z < run_lengths[i]:
							output = config.simulation_model(simulation_jobs[i], worker_bit_generator)
							sums[i] += output
							output_history.append(output)
							z += 1

						est_variances[i] = np.var(output_history)

					comm.send(np.append(sums, est_variances), dest = master_rank)

				# after n0
				else:
					for i in range(num_jobs):
						z = 0

						while z < run_lengths[i]:
							output = config.simulation_model(simulation_jobs[i], worker_bit_generator)
							sums[i] += output
							z += 1

					comm.send(sums, dest = master_rank)

			elif num_jobs == 0:
				# print("Worker " + str(rank) + " finished.")
				return 0 

#############################
######## Execution ##########
#############################

if rank == master_rank:
	print("Running trial " + str(trial))
biPASS(trial)
