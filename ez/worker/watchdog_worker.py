# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray

@ray.remote
class WatchdogServer(object):
    def __init__(self):
        self.reanalyze_batch_count = 0
        self.training_step_count = 0

    def increase_reanalyze_batch_count(self):
        self.reanalyze_batch_count += 1

    def get_reanalyze_batch_count(self):
        return self.reanalyze_batch_count

    def increase_training_step_count(self):
        self.training_step_count += 1

    def get_training_step_count(self):
        return self.training_step_count


# ======================================================================================================================
# watchdog server
# ======================================================================================================================
def start_watchdog_server(manager):
    """
    Start a watchdog server. Call this method remotely.
    """
    watchdog_server = WatchdogServer.remote()
    print('[Watchdog Server] Watchdog server initialized.')
    return watchdog_server


@ray.remote
def start_watchdog_worker(watchdog_server):
    """
    Start a watchdog that monitors training statistics. Call this method remotely.
    """

    # start watching statistics
    last_batch_count = 0
    last_training_step_count = 0
    while True:

        # watchdog
        time.sleep(10)
        batch_count = ray.get(watchdog_server.get_reanalyze_batch_count.remote())
        training_step_count = ray.get(watchdog_server.get_training_step_count.remote())
        last_batch_count = batch_count
        last_training_step_count = training_step_count