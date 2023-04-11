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
num_total_workers = size - 1
master_rank = size - 1

status = MPI.Status()


class Master():

    def __init__(self):

        num_systems = len(config.true_means)

        self.good_systems = tuple(np.where(np.array(config.true_means) >= config.true_standard)[0])
        self.num_systems = num_systems

        self.running_sums = np.full(num_systems, 0.0)
        self.reps = np.full(num_systems, 0.0)
        self.run_lengths = np.full(num_systems, 0.0)

        self.slicepoints = ()

        self.standard = config.init_standard

        if config.known_variance == True:
            self.est_variances = np.array(config.true_variances)
            self.scaling_factors = self.est_variances
        else:
            self.est_variances = np.full(num_systems, np.inf)
            self.scaling_factors = np.full(num_systems, np.inf)

        self.contenders = np.arange(num_systems)
        self.good_contenders = np.array(num_systems)

        self.num_working_workers = num_total_workers

        self.cycle = 0

        self.wallclock_time = time.process_time()

    def compute_run_lengths(self):
        contenders = self.contenders

        self.run_lengths[contenders] = np.full(len(contenders),
                                               config.run_length(self.reps[contenders][0]))

    def compute_slicepoints(self):

        num_contenders = len(self.contenders)
        num_working_workers = self.num_working_workers

        # (num_working_workers - leftover) workers get (base_assignment) systems
        # (leftover) workers are assigned (base_assignment + 1) number of systems
        base_assignment = int(np.floor(num_contenders / float(num_working_workers)))
        leftover = num_contenders % num_working_workers

        # slicepoints is a tuple of length (num_working_workers + 1)
        # slicepoints are used to "slice" the set of contenders
        # worker i is assigned to simulate contenders[slicepoints[worker_id]]
        #	through contenders[slicepoints[worker_id]+1] (consecutively)
        slicepoints = np.append([0],
                                np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                                    np.full(num_working_workers - leftover, base_assignment))))

        self.slicepoints = slicepoints

    def send_worker_jobs(self):

        slicepoints = self.slicepoints

        for worker_id in range(self.num_working_workers):
            worker_assignment_indices = self.contenders[slicepoints[worker_id]:slicepoints[worker_id + 1]]
            data_to_send = tuple((worker_assignment_indices,
                                  self.run_lengths[worker_assignment_indices],
                                  self.reps[worker_assignment_indices]))
            comm.send(data_to_send, dest=worker_id)

    def receive_worker_jobs(self):

        slicepoints = self.slicepoints

        num_pending_messages = self.num_working_workers

        while num_pending_messages > 0:

            data = comm.recv(source=MPI.ANY_SOURCE, status=status)
            worker_id = status.Get_source()

            worker_assignment_indices = self.contenders[slicepoints[worker_id]:slicepoints[worker_id + 1]]

            self.running_sums[worker_assignment_indices] += data[:slicepoints[worker_id + 1] - slicepoints[worker_id]]
            self.reps[worker_assignment_indices] += self.run_lengths[worker_assignment_indices]

            # n0 round: estimate variances
            if self.cycle == 1 and config.known_variance == False:
                self.est_variances[worker_assignment_indices] = data[
                                                                slicepoints[worker_id + 1] - slicepoints[worker_id]:]

            num_pending_messages -= 1

    def update_scaling_factors(self):

        for job_id in range(self.num_systems):
            if config.scaling_type == "custom":
                self.scaling_factors[job_id] = self.est_variances[job_id]
            elif config.scaling_type == "pooled":
                self.scaling_factors[job_id] = np.average(self.est_variances)

    def update_standard(self):
        self.standard = config.update_standard(self.running_sums, self.reps, self.contenders)

    def check_elimination(self):

        contenders = self.contenders

        contenders_reps = self.reps[contenders]
        contenders_scaling_factors = self.scaling_factors[contenders]

        self.contenders = contenders[(np.divide(self.running_sums[contenders] - contenders_reps * self.standard,
                                                contenders_scaling_factors) > config.boundary_function(
            np.divide(contenders_reps, contenders_scaling_factors)))]

        self.good_contenders = np.intersect1d(self.contenders, self.good_systems)

    def terminate_excess_workers(self):

        for worker_id in range(len(self.contenders), self.num_working_workers):
            comm.send(np.array([]), dest=worker_id)

    def terminate_remaining_workers(self):

        for worker_id in range(self.num_working_workers):
            comm.send(np.array([]), dest=worker_id)

    def write_summary_csv(self):
        sample_means = np.divide(self.running_sums, self.reps)
        new_summary_row = pd.DataFrame([[np.argmax(sample_means),
                                         np.max(sample_means),
                                         self.standard,
                                         len(self.contenders),
                                         len([system for system in self.good_systems if system not in self.contenders]),
                                         self.cycle,
                                         np.sum(self.reps),
                                         time.process_time() - self.wallclock_time]],
                                       columns=["Best",
                                                "Best Sample Mean",
                                                "Standard",
                                                "Contenders",
                                                "False Eliminations",
                                                "Cycles",
                                                "Total Effort",
                                                "Time"])
        new_summary_row.to_csv("biPASS_sync_master_summary_" + str(output_file_name) + ".csv", mode="a",
                               header=(trial == 0))

    def write_termination_csv(self):
        np.savetxt(str(output_file_name) + "_reps.csv", self.reps, delimiter=",")
        np.savetxt(str(output_file_name) + "_means.csv", np.divide(self.running_sums, self.reps), delimiter=",")
        np.savetxt(str(output_file_name) + "_variances.csv", self.est_variances, delimiter=",")


class Worker:

    def __init__(self, rank):

        self.rank = rank
        self.cycle = 0

        self.worker_bit_generator = np.random.MT19937(config.base_bit_generator_seed)
        self.initialize_bit_generator()

    def initialize_bit_generator(self):

        base_bit_generator = np.random.MT19937(0).jumped(int((trial + 1) * num_total_workers))

        for worker_id in range(num_total_workers):
            if self.rank == worker_id:
                self.worker_bit_generator = base_bit_generator
                base_bit_generator = base_bit_generator.jumped()


def biPASS():
    ##########################
    ######## MASTER ##########
    ##########################

    if rank == master_rank:

        master = Master()

        master.cycle = 0

        while master.cycle < config.num_cycles and len(master.contenders) > num_total_workers:

            master.cycle += 1

            master.terminate_excess_workers()

            master.compute_run_lengths()

            master.compute_slicepoints()

            master.send_worker_jobs()

            master.receive_worker_jobs()

            if master.cycle == 1:
                master.update_scaling_factors()

            if config.init_standard == -np.inf:
                master.update_standard()

            master.check_elimination()

        master.terminate_remaining_workers()

        master.write_summary_csv()

        master.write_termination_csv()

        return 0

    ##########################
    ######## WORKER ##########
    ##########################

    else:  # if worker

        worker = Worker(rank)

        worker.cycle = 0

        while worker.cycle <= config.num_cycles:

            worker.cycle += 1

            data_received = comm.recv(source=master_rank)

            if len(data_received) == 0:
                return 0

            else:
                num_jobs = len(data_received[0])

                sums = np.zeros(num_jobs)

                estimate_variances_required = (worker.cycle == 1 and config.known_variance == False)

                if estimate_variances_required:
                    est_variances = np.zeros(num_jobs)

                simulation_jobs = data_received[0]
                run_lengths = data_received[1]
                reps = data_received[2]

                for i in range(num_jobs):

                    z = 0

                    if estimate_variances_required:
                        output_history = []

                    while z < run_lengths[i]:
                        output = config.simulation_model(simulation_jobs[i], worker.worker_bit_generator, reps[i])
                        sums[i] += output
                        z += 1

                        if estimate_variances_required:
                            output_history.append(output)

                    if estimate_variances_required:
                        est_variances[i] = np.var(output_history)

                if estimate_variances_required:
                    data_to_send = np.append(sums, est_variances)
                else:
                    data_to_send = sums

                comm.send(data_to_send, dest=master_rank)


#############################
######## Execution ##########
#############################

if rank == master_rank:
    print("Running trial " + str(trial))
biPASS()
