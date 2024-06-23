# --------------------------------------------------------------------------------
# Author: Loping151
# GitHub: https://github.com/Loping151/pytools151
# Description: This repository contains a collection of Python tools designed to
#              enhance productivity and simplify various tasks. The code is open
#              for use and can be freely used, modified, and distributed.
# License: MIT License - Feel free to use and modify the code as you wish.
# --------------------------------------------------------------------------------
import time


class Timer:
    def __init__(self, auto_start=True, log_file="time.log", time_func=time.perf_counter):
        """
        auto_start: bool, default True
            Whether to start the timer automatically.
        log_file: str, default "time.log"
            The file to log the time. Will also create a csv file with the same name.
        time_func: function, default time.perf_counter
            The function to get the current time.
        """
        self.log_file = log_file
        self.log_csv = log_file.split('.')[-2]+'.csv'
        print(f"Log file: {self.log_file} and {self.log_csv}")
        self.time_func = time_func
        self.start_time = None
        self.last_time = None

        self.interval_time = {}
        self.interval_pool = {}

        if auto_start:
            self.start()

    def start(self):
        """Start the timer."""
        self.start_time = self.time_func()
        self.last_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"New timer start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def refresh(self):
        """Equal to an empty lap."""
        self.last_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"Timer refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def lap(self, tag=''):
        """Record a lap."""
        end_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"Lap: {tag}: {end_time - self.last_time}\n")
        self.last_time = end_time

    def start_interval(self, name):
        """Start an interval with a name."""
        if self.interval_time.get(name, -1) >= 0:
            print(f"Interval {name} is not lapped stop. This should not happen.")
            return
        self.interval_time[name] = self.time_func()

    def stop_interval(self, name):
        """Stop an interval with a name. Should pair with start_interval."""
        if self.interval_time.get(name, -1) < 0:
            print(f"Interval {name} is not started. This should not happen.")
            return
        end_time = self.time_func()
        if name not in self.interval_pool:
            self.interval_pool[name] = []
        self.interval_pool[name].append(end_time - self.interval_time[name])
        with open(self.log_file, "a") as f:
            f.write(f"Interval: {name} for the {len(self.interval_pool[name])}-th time: {end_time - self.interval_time[name]}\n")
        self.interval_time[name] = -1

    def __write_csv(self):
        with open(self.log_csv, "a") as f:
            f.write("Timer start at: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
            f.write("Name, Count, Total/s, Average/s\n")
            for name, times in self.interval_pool.items():
                f.write(f"{name}, {len(times)}, {sum(times)}, {sum(times)/len(times)}\n")
            f.write("\n")
            f.write("Details:\n")
            f.write("Name, Count, Total/s, Average/s\n")
            for name, times in self.interval_pool.items():
                for i, t in enumerate(times):
                    f.write(f"{name}, {i+1}, {t}, {t/(i+1)}\n")
    
    def stop(self):
        """Stop the timer and write the log. Timer will be reset."""
        end_time = self.time_func()
        self.__write_csv()
        with open(self.log_file, "a") as f:
            f.write(f"Timer end after: {end_time - self.start_time}\n")
            f.write("Timer reset\n")
        self.start_time = None
        self.last_time = None
        self.interval_time = {}
        self.interval_pool = {}


if __name__ == "__main__":
    import numpy as np
    
    timer = Timer()

    timer.start()

    for i in range(1000000):
        a = np.sqrt(1000)
    timer.lap("sqrt 1000000 times from start")

    time.sleep(1)
    timer.refresh()

    for i in range(2000000):
        a = np.sqrt(1000)
    timer.lap("sqrt 2000000 times from refresh")

    for i in range(10):
        timer.start_interval("sqrt 1000 times and square 2000 times")
        for _ in range(1000):
            a = np.sqrt(1000)
        timer.start_interval("square 2000 times and matmul 4000 times") # ovrelap between different name of intervals is allowed
        for _ in range(2000):
            b = np.square(2000)
        timer.stop_interval("sqrt 1000 times and square 2000 times")
        for _ in range(4000):
            c = np.matmul(np.ones((100, 100)), np.ones((100, 100)))
        timer.stop_interval("square 2000 times and matmul 4000 times")

    timer.stop()