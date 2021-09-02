class EsriGrid(object):
    def __init__(self, ncols, nrows, xllcorner, yllcorner, cellsize, grid_data, filepath, NODATA_VALUE=-9999):
        self.ncols = ncols
        self.nrows = nrows
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.NODATA_VALUE = NODATA_VALUE
        self.grid_data = grid_data
        self.filepath = filepath

    def read_file(self):
        f = open(self.filepath, "r")
        return f

    def write_file(self):
        f = open(self.filepath, "w")

        # create file header
        f.write("ncols {}\n".format(self.ncols))
        f.write("nrows {}\n".format(self.nrows))
        f.write("xllcorner     {}\n".format(self.xllcorner))
        f.write("yllcorner     {}\n".format(self.yllcorner))
        f.write("cellsize      {}\n".format(self.cellsize))
        f.write("NODATA_value  {}\n".format(self.NODATA_VALUE))

        # write data rows
        for row in range(self.nrows):
            for col in range(self.ncols):
                f.write("{} ".format(self.grid_data[row, col] if self.grid_data[row, col] != 0 else self.NODATA_VALUE ))
            # new row
            f.write("\n")

        # close file
        f.close()


def g2height(v, ori_Min=0, ori_Max=255, targetMin=0.0, targetMax=150.0):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv
