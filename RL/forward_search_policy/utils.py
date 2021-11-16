import math

class MovingAvgCalculator():
    def __init__(self, window_size):
        self.window_size = window_size
        self.num_added = 0
        self.window = [0.0 for _ in range(window_size)]

        self.avg = 0.0
        self.var = 0.0
        self.last_std = 0.0

    def update(self, value):
        idx = self.num_added % self.window_size
        old_value = self.window[idx]
        self.window[idx] = value
        self.num_added += 1

        old_avg = self.avg
        if self.num_added <= self.window_size:
            delta = value - old_avg
            self.avg += delta / self.num_added
            self.var += delta * (value - self.avg)
        else:
            delta = value - old_value
            self.avg += delta / self.window_size
            self.var += delta * ((value - self.avg) + (old_value - old_avg))

        if self.num_added <= self.window_size:
            if self.num_added == 1:
                variance = 1
            else:
                variance = self.var / (self.num_added - 1)
        else:
            variance = self.var / self.window_size

        try:
            std = math.sqrt(variance)
            if math.isnan(std):
                std = 0.1
        except:
            std = 0.1

        self.last_std = std

        return self.avg, std

    def get_std(self):
        return self.last_std