from astropy.table import Table
import pkg_resources

# load Gianninas 2013 tables into memory
bands = ('u', 'g', 'r', 'i', 'z')
tables = dict()
for band in bands:
    filename = 'data/ld_coeffs_{}.txt'.format(band)
    # get installed locations
    filename = pkg_resources.resource_filename('pylcurve', filename)
    tables[band] = Table.read(filename, format='cds')


# should create interpolator class here so other modules can import and use
# without overhead of creating interpolation each time
