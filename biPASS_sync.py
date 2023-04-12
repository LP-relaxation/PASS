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

# initialize MPI objects
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_total_workers = size - 1
master_rank = size - 1

status = MPI.Status()


class Master:

    def __init__(self, trial):

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

        self.num_total_workers = num_total_workers

        self.cycle = 0

        self.wallclock_start_time = time.process_time()

        self.use_rinott = config.use_rinott

    def compute_run_lengths(self):
        contenders = self.contenders

        self.run_lengths[contenders] = np.full(len(contenders),
                                               config.run_length(self.reps[contenders][0]))

    def compute_slicepoints(self):

        num_contenders = len(self.contenders)
        num_total_workers = self.num_total_workers

        # (num_total_workers - leftover) workers get (base_assignment) systems
        # (leftover) workers are assigned (base_assignment + 1) number of systems
        base_assignment = int(np.floor(num_contenders / float(num_total_workers)))
        leftover = num_contenders % num_total_workers

        # slicepoints is a tuple of length (num_total_workers + 1)
        # slicepoints are used to "slice" the set of contenders
        # worker i is assigned to simulate contenders[slicepoints[worker_id]]
        #	through contenders[slicepoints[worker_id]+1] (consecutively)
        slicepoints = np.append([0],
                                np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                                    np.full(num_total_workers - leftover, base_assignment))))

        self.slicepoints = slicepoints

    def send_worker_jobs(self):

        slicepoints = self.slicepoints

        for worker_id in range(self.num_total_workers):
            worker_assignment_indices = self.contenders[slicepoints[worker_id]:slicepoints[worker_id + 1]]
            data_to_send = tuple((worker_assignment_indices,
                                  self.run_lengths[worker_assignment_indices],
                                  self.reps[worker_assignment_indices]))
            comm.send(data_to_send, dest=worker_id)

    def receive_worker_jobs(self):

        slicepoints = self.slicepoints

        num_pending_messages = self.num_total_workers

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

    def terminate_workers(self):

        for worker_id in range(self.num_total_workers):
            comm.send(np.array([]), dest=worker_id)

    def write_biPASS_cycle_csv(self):
        sample_means = np.divide(self.running_sums, self.reps)
        new_summary_row = pd.DataFrame([
            [np.argmax(sample_means),
             np.max(sample_means),
             self.standard,
             len(self.contenders),
             len([system for system in self.good_systems if system not in self.contenders]),
             self.cycle,
             np.sum(self.reps),
             time.process_time() - self.wallclock_start_time]],
            columns=["Best",
                     "Best Sample Mean",
                     "Standard",
                     "Contenders",
                     "False Eliminations",
                     "Cycles",
                     "Total Effort",
                     "Time"])
        new_summary_row.to_csv("biPASS_cycle_master_" + str(output_file_name) + "_" + str(trial) + ".csv", mode="a",
                               header=(trial == 0 and self.cycle == 1))

    def write_biPASS_termination_csv(self):

        filename_prefix = "biPASS_summary_" + str(output_file_name)

        np.savetxt(filename_prefix + "_reps_" + str(trial) + ".csv", self.reps, delimiter=",")
        np.savetxt(filename_prefix + "_means_" + str(trial) + ".csv", np.divide(self.running_sums, self.reps), delimiter=",")
        np.savetxt(filename_prefix + "_variances_" + str(trial) + ".csv", self.est_variances, delimiter=",")

    def compute_rinott_constant(self):

        num_contenders = len(self.contenders)
        sample_size_quantile_est = int(1e6)
        n0 = config.run_length(0)

        Z = np.random.normal(size=(sample_size_quantile_est, num_contenders - 1))
        Y = np.random.chisquare(df=n0 - 1, size=(sample_size_quantile_est, num_contenders - 1))
        C = np.random.chisquare(df=n0 - 1, size=sample_size_quantile_est)
        C = np.reshape(C, (len(C), 1))
        Cmat = np.repeat(C, num_contenders - 1, axis=1)
        denom = np.sqrt((n0 - 1) * (1 / Y + 1 / Cmat))
        H = np.sort(np.max(Z * denom, axis=1))
        rinott_constant = np.quantile(H, 1 - config.alpha_rinott)

        return rinott_constant

    def compute_rinott_run_lengths(self, rinott_constant):

        contenders = self.contenders

        self.run_lengths[contenders] = self.est_variances[contenders] * \
                                       (rinott_constant / config.IZ_param) ** 2

    def reset_running_sums(self):
        self.running_sums = np.full(self.num_systems, 0.0)

    def reset_reps(self):
        self.reps = np.zeros(self.num_systems)

    def write_rinott_termination_csv(self):

        filename_prefix = "rinott_summary_" + str(output_file_name)

        running_sums = self.running_sums
        reps = self.reps
        contenders = self.contenders

        np.savetxt(filename_prefix + "_reps_" + str(trial) + ".csv", self.reps, delimiter=",")
        np.savetxt(filename_prefix + "_means_" + str(trial) + ".csv", np.divide(running_sums[contenders], reps[contenders]), delimiter=",")

        max_contenders_ix = np.argmax(np.divide(running_sums[contenders], reps[contenders]))
        best_ix = contenders[max_contenders_ix]
        best_mean = np.divide(running_sums[contenders], reps[contenders])[max_contenders_ix]

        solution_message = "Best solution is " + str(best_ix) + " with estimated mean " + str(best_mean)

        with open(filename_prefix + "_good_selection.txt", 'w', encoding='utf-8') as f:
            f.write(solution_message + "\n")

        f.close()

        print(solution_message)


class Worker:

    def __init__(self, rank, trial):

        self.worker_bit_generator = np.random.MT19937(config.base_bit_generator_seed)
        self.rank = rank
        self.cycle = 0

        self.initialize_bit_generator()

    def initialize_bit_generator(self):

        base_bit_generator = np.random.MT19937(config.base_bit_generator_seed).jumped(
            int((trial + 1) * num_total_workers))

        for worker_id in range(num_total_workers):
            if self.rank == worker_id:
                self.worker_bit_generator = base_bit_generator
                base_bit_generator = base_bit_generator.jumped()

    def simulate_system(self, system_id, run_length, current_num_reps, estimate_variance=False):

        z = 0
        running_sum = 0

        if estimate_variance:

            output_history = []

            while z < run_length:
                z += 1
                output = config.simulation_model(system_id,
                                                 self.worker_bit_generator,
                                                 int(current_num_reps + z))
                running_sum += output
                output_history.append(output)

            est_variance = np.var(output_history)

            return running_sum, est_variance

        else:

            while z < run_length:
                z += 1
                output = config.simulation_model(system_id,
                                                 self.worker_bit_generator,
                                                 int(current_num_reps + z))
                running_sum += output

            return running_sum

    def finish_jobs(self, data_from_master):

        num_jobs = len(data_from_master[0])

        running_sums = np.zeros(num_jobs)

        simulation_jobs = data_from_master[0]
        run_lengths = data_from_master[1]
        reps = data_from_master[2]

        estimate_variances_required = (self.cycle == 1 and config.known_variance == False)

        if estimate_variances_required:
            est_variances = np.zeros(num_jobs)
            for i in range(num_jobs):
                running_sums[i], est_variances[i] = self.simulate_system(simulation_jobs[i],
                                                                         run_lengths[i],
                                                                         reps[i],
                                                                         True)
            finished_simulation_data = np.append(running_sums, est_variances)

        else:
            for i in range(num_jobs):
                running_sums[i] = self.simulate_system(simulation_jobs[i],
                                                       run_lengths[i],
                                                       reps[i],
                                                       False)
            finished_simulation_data = running_sums

        return finished_simulation_data

    @staticmethod
    def send_data(data):
        comm.send(data, dest=master_rank)

    @staticmethod
    def receive_data():
        return comm.recv(source=master_rank)


def biPASS(trial):
    ##########################
    ######## MASTER ##########
    ##########################

    if rank == master_rank:

        master = Master(trial)

        master.cycle = 0

        while master.cycle < config.num_cycles and len(master.contenders) > master.num_total_workers:

            master.cycle += 1

            master.compute_run_lengths()

            master.compute_slicepoints()

            master.send_worker_jobs()

            master.receive_worker_jobs()

            if master.cycle == 1:
                master.update_scaling_factors()

            if config.init_standard == -np.inf:
                master.update_standard()

            master.check_elimination()

            if config.cycle_output:
                master.write_biPASS_cycle_csv()

        master.write_biPASS_termination_csv()

        if master.use_rinott:

            rinott_constant = master.compute_rinott_constant()

            master.compute_rinott_run_lengths(rinott_constant)

            master.reset_running_sums()
            master.reset_reps()

            master.compute_slicepoints()

            master.send_worker_jobs()

            master.receive_worker_jobs()

            master.write_rinott_termination_csv()

        master.terminate_workers()

        return 0

    ##########################
    ######## WORKER ##########
    ##########################

    else:  # if worker

        worker = Worker(rank, trial)

        worker.cycle = 0

        while worker.cycle <= config.num_cycles:

            worker.cycle += 1

            data_from_master = worker.receive_data()

            if len(data_from_master) == 0:
                return 0

            else:
                finished_simulation_data = worker.finish_jobs(data_from_master)
                worker.send_data(finished_simulation_data)


#############################
######## Execution ##########
#############################

if rank == master_rank:
    print("Running trial " + str(trial))
biPASS(trial)
