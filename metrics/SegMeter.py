import numpy as np


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.    
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Resets the meter to default settings."""
        pass

    def update(self, value):
        """Log a new value to the meter.

        Args:
            value: Next result to include.

        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    """Average value meter stores mean and standard deviation
    for population of input values. Meter updates are applied online, 
    one value for each update. Values are not cached, only the last added.

    Examples::
        # >>> Initialize a meter to record loss
        # >>>     losses = AverageValueMeter()
        # >>> Update meter after every minibatch update
        # >>>     losses.update(loss_value, batch_size)
    """

    def __init__(self):
        """Constructor method for the ``AverageValueMeter`` class."""
        super(AverageValueMeter, self).__init__()
        self.n = 0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

    def add(self, value) -> None:
        """Add a new observation.

        Updates of mean and std are going online, with
        `Welford's online algorithm
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.

        Args:
            value (float): value for update,
                can be scalar number or PyTorch tensor

        .. note::
            Because of algorithm design,
            you can update meter values with only one value a time.
        """
        self.val = value
        self.n += 1

        if self.n == 1:
            self.mean = 0.0 + value  # Force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        """Returns meter values.

        Returns:
            mean (float): Mean that has been updated online.
            std (float): Standard deviation that has been updated online.
        """
        return self.mean, self.std

    def reset(self):
        """Resets the meter to default settings."""
        self.n = 0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
