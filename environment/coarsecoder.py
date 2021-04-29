import numpy as np
import pandas as pd


class CoarseCoder:

    def __init__(self, config, pos_range=(-1.2, 0.6), velocity_range=(-0.07, 0.07)):
        """
        Defines the parameters the encoder will use when encoding.
        :param granularity: How many 'buckets' to sort each axis into as tuple (pos-granularity, vel-granularity)
        :param pos_overlap: The overlap between position buckets
        :param velocity_overlap: The overlap between velocity buckets
        :param pos_range: The range for the position (as tuple)
        :param velocity_range: The range for the velocity (as tuple)
        """
        self.pos_overlap = config["pos_overlap"]
        self.velocity_overlap = config["velocity_overlap"]
        self.granularity = config["granularity"]
        self.velocity_range = velocity_range
        self.pos_range = pos_range

    def get_coarse_encoding(self, pos, velocity):
        """
        Gets a coarse encoded numpy array for the supplied position and velocity
        :param pos: The position
        :param velocity: The velocity
        :return: The coarse encoded numpy array
        """
        if self.pos_range[0] > pos > self.pos_range[1] or self.velocity_range[0] > velocity > self.velocity_range[1]:
            raise AttributeError("Pos ({}) and velocity ({}) is outside ranges for coder, {} & {}".format(
                pos, velocity, self.pos_range, self.velocity_range
            ))

        # Raise error if overlap between buckets is larger than the range of the bucket
        range_pos = (abs(self.pos_range[0]) +
                     abs(self.pos_range[1]))/self.granularity[0]
        range_vel = (
            abs(self.velocity_range[0]) + abs(self.velocity_range[1]))/self.granularity[1]
        if pos_overlap > range_pos or velocity_overlap > range_vel:
            raise ValueError("Overlap of buckets is larger than range of the bucket. Ranges: position: {} & velocity: {}. Overlaps: position: {} & velocity: {}".format(
                range_pos, range_vel, self.pos_overlap, self.velocity_overlap
            ))

        coarse_array = self._create_empty_encoding_df()
        for vel_bin, row in coarse_array.iterrows():
            for pos_bin, val in row.iteritems():
                if self._value_in_bin(velocity, vel_bin) and self._value_in_bin(pos, pos_bin):
                    coarse_array.loc[[vel_bin], [pos_bin]] = 1

        return coarse_array.to_numpy()

    @staticmethod
    def _value_in_bin(val, b):
        """
        Returns True if the value is in the bin, else False 
        :param val: The value
        :param b: The bin (start_val, end_val)
        :return: True if value in bin, else False
        """
        return b[0] <= val <= b[1]

    def _create_empty_encoding_df(self):
        """
        Creates a pandas data-frame (all zeros) to use for coarse encoding.
        The rows and columns are assigned headers that correspond to the bins they are representing.
        :return: The data-frame
        """
        encoding_array = []

        # Create the bins for position
        pos_starts = np.linspace(
            self.pos_range[0], self.pos_range[1], self.granularity[0] + 1)[:-1]
        pos_ends = np.array(np.linspace(self.pos_range[0], self.pos_range[1], self.granularity[0] + 1) +
                            self.pos_overlap)[1:]
        pos_bins = self._create_axis_bins(pos_starts, pos_ends)
        # Create the bins for velocity
        vel_starts = np.linspace(
            self.velocity_range[0], self.velocity_range[1], self.granularity[1] + 1)[:-1]
        vel_ends = np.array(np.linspace(self.velocity_range[0], self.velocity_range[1], self.granularity[1] + 1) +
                            self.velocity_overlap)[1:]
        vel_bins = self._create_axis_bins(vel_starts, vel_ends)

        # Create the array with all zeros
        for i in range(self.granularity[0]):
            row = []
            for j in range(self.granularity[1]):
                row.append(0)
            encoding_array.append(row)

        df = pd.DataFrame(encoding_array, index=vel_bins, columns=pos_bins)
        return df

    @staticmethod
    def _create_axis_bins(starts, ends):
        """
        Given the start and end of a list of bins, returns the ranges of all the bins.
        :param starts: The starts of the bins
        :param ends: The ends of the bins
        :return: The ranges of all the bins
        """
        bins = []
        for i in range(len(starts)):
            bins.append((starts[i], ends[i]))
        return bins


# if __name__ == '__main__':
#    a = CoarseCoder(0.05, 0.05, (10, 10))
 #   print(a.get_coarse_encoding(0.5, 0.07))
