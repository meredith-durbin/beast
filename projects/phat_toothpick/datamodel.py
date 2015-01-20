""" Data Model interface v2.0

Includes artificial star tests (ASTs) and related function to generate the
noise model

The noise model is then applied to the models while only data measurements are
used in constrasts with using measurement uncertainties.

with limited quantity units handling to help both reading and robustness.

..note::

    By default units on returned values from function calls are turned off to
    avoid breaking possible other scripts.
"""
import numpy as np

# BEAST imports
from beast.core import stellib
from beast.core import extinction
from beast.core.observations import Observations
from beast.core.vega import Vega
from beast.external.ezunits import unit

#---------------------------------------------------------
# User inputs                                   [sec:conf]
#---------------------------------------------------------
# Parameters that are required to make models
# and to fit the data
#---------------------------------------------------------
project = 'b15_jan15_test'

filters = ['HST_WFC3_F275W', 'HST_WFC3_F336W', 'HST_ACS_WFC_F475W',
           'HST_ACS_WFC_F814W', 'HST_WFC3_F110W', 'HST_WFC3_F160W']

# observations
#obsfile = 'data/b15_data_test.fits'
obsfile = 'data/b15_4band_det_27_A.fits'

# AST files (single camera ASTs)
uvastfile = 'data/fake_stars_b15_27_uv.fits'
optastfile = 'data/fake_stars_b15_27_opt.fits'
irastfile = 'data/fake_stars_b15_27_ir.fits'
astfile = 'data/fake_stars_b15_27_all.hd5'

# name for noise model
noisefile = project + '/' + project + '_noisemodel.hd5'

# absflux calibration covariance matrix for PHAT specific filters

absflux_a_matrix = np.array(
    [[1.19, 1.11, 0.74, 0.53, 0.30, 0.22],
     [1.11, 1.04, 0.74, 0.57, 0.41, 0.36],
     [0.74, 0.74, 0.71, 0.69, 0.68, 0.67],
     [0.53, 0.57, 0.69, 0.74, 0.77, 0.77],
     [0.30, 0.41, 0.68, 0.77, 0.84, 0.86],
     [0.22, 0.36, 0.67, 0.77, 0.86, 0.88]])

absflux_a_matrix = (absflux_a_matrix*0.01)**2.

# distance to M31
distanceModulus = 24.47 * unit['mag']

### Stellar grid definition

# log10(Age) -- [min,max,step] to generate the isochrones
logt = [6.0, 10.13, 0.15]

#note: Mass is not sampled, use the isochrone def instead.

#Metallicity
z = [0.03, 0.019, 0.008, 0.004]
#z = 0.019

# Isochrone CMD version (2.3 for Girardi et al. (2010) or 2.7 for PARSECv1.2S)
trackVersion = 2.3

# Stellar Atmospheres library definition
osl = stellib.Tlusty() + stellib.Kurucz()

################

### Dust extinction grid definition
extLaw = extinction.RvFbumpLaw()

# A(V): dust column
avs = [0.0, 10.055, 0.15]
#avs = [0.0, 10.055, 0.5]

# R(V): dust average grain size
rvs = [2.0, 6.1, 0.5]
#rvs = [3.0,3.0,1.0]

# fbump (should be f_A): mixture factor between "MW" and "SMCBar" extinction curves
fbumps = [0., 1.01, 0.25]
#fbumps = [1.0,1.0, 0.25]

################

## ..note::
##      in the above grid definitions the upper limit makes sure the last point of interest is
##      included


class PHATFluxCatalog(Observations):
    """PHAT 6 filter photometry
    This class implements a direct access to the PHAT measured fluxes.

    ..note::
        it does not implement uncertainties as in this model, the noise is
        given through artificial star tests
    """
    def __init__(self, inputFile, distanceModulus=distanceModulus, filters=filters):
        """ Construct the interface """
        desc = 'PHAT star: %s' % inputFile
        Observations.__init__(self, inputFile, distanceModulus, desc=desc)
        self.setFilters( filters )
        #some bad values smaller than expected
        # in RATE = flux units
        self.setBadValue(6e-30)

        #hard code mapping directly with the interface to PHAT
        for k in filters:
            self.data.set_alias(k, k.split('_')[-1].lower() + '_rate')

    def getFlux(self, num, units=False):
        """returns the absolute flux of an observation from the number of
        counts

        Parameters
        ----------
        num: int
            index of the star in the catalog to get measurement from

        units: bool
            if set returns the fluxes with a unit capsule

        Returns
        -------
        flux: ndarray[dtype=float, ndim=1]
            Measured integrated flux values throughout the filters in erg/s/cm^2
        """

        # case for using '_RATE' result instead of '_VEGA'
        d = self.data[num]
        flux = np.array([ d[self.data.resolve_alias(ok)] for ok in self.filters ]) * self.vega_flux
        #flux = Observations.getFlux(self, num) * self.vega_flux
        
        # catalog uses _VEGA which are vega magnitudes.
        #d = self.data[num]
        #flux = 10 ** (-0.4 * np.array([ d[self.data.resolve_alias(ok)] for ok in self.filters ])) * self.vega_flux

        if units is True:
            return flux * unit['erg/s/cm**2']
        else:
            return flux

    def setFilters(self, filters):
        """ set the filters and update the vega reference for the conversions

        Parameters
        ----------
        filters: sequence
            list of filters using the internally normalized namings
        """
        self.filters = filters

        #Data "rates" are normalized to Vega already, fits are not using vega

        # for optimization purpose: pre-compute
        #   getting vega mags, require to open and read the content of one file.
        #   since getObs, calls getFlux, for each star you need to do this expensive
        #   op.
        with Vega() as v:
            _, vega_flux, _ = v.getFlux(filters)

        self.vega_flux = vega_flux


def get_obscat(obsfile=obsfile, distanceModulus=distanceModulus,
               filters=filters, *args, **kwargs):
    """ Function that generates a data catalog object with the correct
    arguments

    Parameters
    ----------
    obsfile: str, optional (default datamodel.obsfile)
        observation file

    distanceModulus: float, optional (default datamodel.distanceModulus)
        distance modulus to correct the data from (in magitude)

    filters: sequence(str), optional, datamodel.filters
        seaquence of filters of the data

    returns
    -------
    obs: PHATFluxCatalog
        observation catalog
    """
    obs = PHATFluxCatalog(obsfile, distanceModulus=distanceModulus, filters=filters)
    return obs