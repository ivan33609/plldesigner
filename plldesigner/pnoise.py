# -*- coding: utf-8 -*-
"""
A Class to work with phase noise frequency data points
"""

from __future__ import (absolute_import, division, print_function)
import numpy as np
from numpy import log10, sqrt, sum
import matplotlib.pyplot as plt
import scipy.interpolate as intp

# functions to handle the units
__funits__ = {
    "dBc/Hz": lambda x: x,
    "rad/sqrt(Hz)": lambda x: 10 * log10(x ** 2 / 2),
    "rad**/Hz": lambda x: 10 * log10(x / 2),
}


class Pnoise(object):
    """
    Phase noise class to manipulate phase noise data in the frequency offset
    format

    Parameters
    ----------
    fm : array_like
        Offset frequency vector
    pn : array_like
        Phase noise values

    """

    def __init__(self, fm, pnfm, fc=1, label=None, units='dBc/Hz'):
        """
        Phase noise from separate vectors for fm and phase noise

        Parameters
        ----------
        fm : array_like
            Offset frequency vector
        pn : array_like
            Phase noise values
        fc : float
            Carrier frequency (Hz)

        Returns
        -------
        pnoise : Pnoise
        """
        self.units = units
        self.label = label
        self._fm = np.asarray(fm)
        self._fc = fc

        # values for point slope approximation
        self.slopes = None

        self.phi_out = None

        self.ldbc = __funits__[units](np.asarray(pnfm))

        # keep the original interpolation data
        self._fmi = np.copy(self._fm)
        self._ldbci = np.copy(self.ldbc)

        self.func_ldbc = lambda fx:\
            __pnoise_interp1d__(self._fmi, self._ldbci, fx)

    @property
    def fm(self):
        return self._fm

    @fm.setter
    def fm(self, fi):
        fi = np.asarray(fi)
        self._fm = fi
        self.ldbc = self.func_ldbc(fi)

    @property
    def fc(self):
        return self._fc

    @fc.setter
    def fc(self, fc):
        assert isinstance(fc, (float, int))
        self.ldbc = self.ldbc + 20*np.log10(fc/self._fc)
        self._ldbci = np.copy(self.ldbc)
        self._fc = fc


    @classmethod
    def with_function(cls, func, label=None):
        """
        Class method for manipulating phase noise values

        Parameters
        ----------
        func : function
            A Python function representing the phase noise and function of
            the frequency offset: ldbc = func(fm) (dBc/Hz)
        label : string
            Name of the phase noise value

        Returns
        -------
        pnoise : Pnoise

        """
        pnoise_class = cls(None, None, label=label, units='dBc/Hz')
        pnoise_class.func_ldbc = func
        pnoise_class.label = label
        return pnoise_class

    @classmethod
    def with_points_slopes(cls, fm, ldbc_fm, slopes, label=None):
        """
        Phase noise with point and slope

        Parameters
        ----------
        fm : array_like
            Array with the offsets frequencies of the phase noise values
        ldbc_fm : array_like
            Array with the phase noise values at the fm frequencies
        slopes : array_like
            Array with slopes of the values that are interpolated (dBc/dec)
        label : str
            Plotting label

        Returns
        -------
        pnoise : Pnoise

        """
        fm = np.asarray(fm)
        ldbc_fm = np.asarray(ldbc_fm)
        slopes = np.asarray(slopes)
        pnoise_class = cls(fm, ldbc_fm, label=label, units='dBc/Hz')
        pnoise_class.slopes = slopes
        pnoise_class.func_ldbc = lambda fi: __pnoise_point_slopes__(
            fm, ldbc_fm, slopes, fi)
        return pnoise_class

    def add_points(self, fx, ldbc):
        """
        Add extra points to a noise object
        """
        assert len(fx) == len(ldbc), 'arrays have to be of equal lenght'

        ldbc = np.array(ldbc)
        # Concatenate  vectors
        ldbc = np.hstack((ldbc, self._ldbci))
        fi = np.hstack((fx, self._fmi))
        # Find repeted values
        fi_x, ix  = np.unique(fi, return_index=True)
        ldbc_x = ldbc[ix]

        # Update the object
        ix = np.argsort(fi_x)
        self._fm = fi_x[ix]
        self.ldbc = ldbc_x[ix]
        # change the interpolation values
        self._fmi = np.copy(fi_x[ix])
        self._ldbci = np.copy(ldbc_x[ix])


    def create_new(self, fx):
        """
        Return a Pnoise clase with with different frequency sampling
        """
        pnoise_class = Pnoise(fx, self.func_ldbc(fx), label=self.label)
        return pnoise_class


    def asymptotic_model(self, fi, slopes, label=None):
        """
        Return a phase noise function made with the asymptotic behavior
        extrapolated at certain frequencies

        Parameters
        ----------
        fi : array_like
            Array with the interpolated values
        slopes: array_like
            Array with the slopes were the noise is to going to be interpolated
        label: str
            Phase noise label

        Returns
        -------
        pnoise_class:
            A new class with noise that behaves asymptotically
        """
        fi = np.asarray(fi)
        slopes = np.asarray(slopes)
        ldbc_fi = self.func_ldbc(fi)
        pnoise_class = Pnoise.with_points_slopes(fi, ldbc_fi, slopes, label=label)
        return pnoise_class


    def plot(self, *args, **kwargs):
        plt.ylabel('$\mathcal{L}$(dBc/Hz)')
        plt.xlabel('$f_m$(Hz)')
        ax = plt.semilogx(self.fm, self.ldbc, label=self.label, *args,
                          **kwargs)
        return ax

    def __add__(self, other):
        """
        Addition of though pnoise components
        """
        try:
            phi2fm = 2 * 10 ** (self.ldbc / 10)
            phi2fm_other = 2 * 10 ** (other.ldbc / 10)
            ldbc_add = 10 * log10((phi2fm + phi2fm_other) / 2)
        except ValueError:
            raise ValueError(
                'Additions is only allowed with vector of equal size'
                )
        add_noise = Pnoise(self.fm, ldbc_add)
        return add_noise

    def __mul__(self, mult):
        """
        Multiplication of noise by a constant
        """
        if type(mult) not in (int, float, np.ndarray):
            raise TypeError('unsupported operand type(s) for mult')
        else:
            if type(mult) in (int, float):
                mult_noise = Pnoise(
                    self.fm, self.ldbc + 10 * log10(mult), label=self.label)
            else:
                try:
                    mult_noise = Pnoise(
                        self.fm, self.ldbc + 10 * log10(mult),
                        label=self.label)
                except ValueError:
                    raise ValueError('Vectors are not of the same length')
            return mult_noise

    def integrate(self, fl=None, fh=None, method='trapz'):
        """
        Returns the integrated phase noise in (rad/rms)

        Parameters
        ----------
        fl : float
            Lower frequency integration limit. Default is: min(fm)
        fh : float
            Higher frequency integration limit. Default is: max(fm)
        method : str
            Integration method used. Default is trapz with logarithmic
            interpolation

        Returns
        -------
        phi_out : float
            The integrated phase in rad
        """

        def gardner(ldbc_ix, fm_ix):
            """
            Gardner integration method

            Parameters
            ----------
            ldbc_ix : array_like
                Phase noise in dBc/Hz
            fm_ix : array_like
                Frequency offset for the phase noise

            Returns
            -------
            phi_sqrt : array_like
                Integrated phase in rads

            """
            lfm = len(ldbc_ix)
            # calculate the slope
            ai = ((ldbc_ix[1:lfm] - ldbc_ix[:lfm - 1]) /
                  (log10(fm_ix[1:lfm]) - log10(fm_ix[:lfm - 1])))
            """ If the slopes are never too big used Gardner method
            In simulations this is not the case """
            bi = (2 * 10 ** (ldbc_ix[:lfm - 1] / 10) * fm_ix[:lfm - 1] **
                  (-ai / 10) /
                  (ai / 10 + 1) * (fm_ix[1:lfm] ** (ai / 10 + 1) -
                                   fm_ix[:lfm - 1] ** (ai / 10 + 1)))
            if ai < 6:
                raise Warning(
                    'Gardner method fail due to large slope changes :\n' +
                    'use trapz method'
                )
            return sqrt(sum(bi))

        def trapz(ldbc_ix, fm_ix):
            """
            Trapezoidal integration of the phase noise

            Parameters
            ----------
            ldbc_ix : array_like
                Phase noise in dBc/Hz

            fm_ix : array_like
                Frequency offset for the phase noise points

            Returns
            -------
            phi_out : array_like
                Integrated phase in rad

            """
            phi_2 = 2 * 10 ** (ldbc_ix / 10)
            return sqrt(np.trapz(phi_2, fm_ix))

        if fl is None:
            fl = min(self.fm)
        if fh is None:
            fh = max(self.fm)
        ix = (self.fm >= fl) & (self.fm <= fh)
        fi = self.fm[ix]
        ldbc_fi = self.ldbc[ix]
        if method == 'trapz':
            self.phi_out = trapz(ldbc_fi, fi)
        elif method == 'Gardner':
            self.phi_out = gardner(ldbc_fi, fi)
        else:
            raise ValueError('Integrating method not implemented')
        return self.phi_out

    def interp1d(self, fi):
        """
        Interpolate the phase noise assuming logarithmic linear behaviro

        Parameters
        ---------
        fi : array_like
            Frequency where the noise is to be interpolated

        Return
        ------
        ldbc : array_like
            The interpolated noise at frequencies fi
        """
        fi = np.asarray(fi)
        ldbc = self.func_ldbc(fi)
        return ldbc


def __pnoise_interp1d__(fm, ldbc_fm, fi):
    """ Interpolate the phase noise assuming logarithmic linear behavior

        Parameters
        ---------
        fm :
        ldbc_fm :

        Returns
        ----------
        ldbc_fi :

    """

    func_intp = intp.interp1d(log10(fm), ldbc_fm, kind='linear')
    ldbc_fi = func_intp(log10(fi))
    return ldbc_fi


def __pnoise_point_slopes__(fm, ldbc_fm, slopes, fi):
    """
    Create a functions that follows the asymptotical behavior of phase noise.


    Parameters
    ----------
    fm : array_like
        Frequency offset of the input points
    ldbc_fm : array_like
        Phase noise in dBc/Hz
    slopes : array_like
        Slopes in dB/dec
    fi : array_like
        Points that are to be calculated

    returns
    -------
    ldbc_fi : array_like
        Interpolated points

    """
    phi2 = 2 * 10 ** (ldbc_fm / 10)
    phi2 = phi2.reshape((1, len(phi2)))
    phi2_matrix = np.repeat(phi2, len(fi), axis=0)

    slopes = np.copy(slopes / 10)
    slopes = slopes.reshape((1, len(slopes)))
    slopes_matrix = np.repeat(slopes, len(fi), axis=0)

    fm = fm.reshape((1, len(fm)))
    fi_matrix = np.repeat(fm, len(fi), axis=0)

    fi = fi.reshape((len(fi), 1))
    fm_matrix = np.repeat(fi, fm.shape[1], axis=1)

    phi2_fm = np.sum(
        phi2_matrix * (fm_matrix / fi_matrix) ** slopes_matrix, axis=1)
    ldbc_fi = 10 * log10(phi2_fm / 2)
    return ldbc_fi


"""
Testing
"""
from numpy.testing import assert_almost_equal
import unittest

class Test_pnoise(unittest.TestCase):
    def test_fc_settler(self):
        fm = np.array([1e3, 1e5, 1e7])
        ldbc = 10 * np.log10(1 / (fm * fm))
        lorentzian = Pnoise(fm, ldbc, fc=1e9, label='Lorentzian')
        lorentzian.fc = 10e9
        assert_almost_equal(lorentzian.ldbc, ldbc + 20*log10(10e9/1e9))

    def test_interp1d_class(self, plot=False):
        fm = np.array([1e3, 1e5, 1e7])
        lorentzian = Pnoise(fm, 10 * np.log10(1 / (fm * fm)), label='Lorentzian')
        val = lorentzian.interp1d(1e6)
        assert_almost_equal(val, -120, 4)
        if plot:
            lorentzian.plot()
            plt.show()


    def test__init__(self, plot=False):
        # Test __init__
        fm = np.logspace(3, 9, 100)
        lorentzian = Pnoise(fm, 10 * np.log10(1 / (fm * fm)), label='Lorentzian')
        white = Pnoise(fm, -120 * np.ones(fm.shape), label='white')
        added = white + lorentzian
        added.label = "addition"
        assert_almost_equal(added.ldbc[0], -60, 4)
        assert_almost_equal(added.ldbc[-1], -120, 4)
        ix, = np.where(fm > 1e6)
        assert_almost_equal(added.ldbc[ix[0]], -117.2822, 4)
        if plot:
            lorentzian.plot()
            white.plot()
            added.plot()
            plt.legend()
            plt.show()


    def test_private_functions(self):
        # test the new
        fi = np.array([1e4, 1e9])
        ldbc_fi = np.array([-40, -150])
        slopes = np.array([-30, -20])
        fm = np.logspace(3, 9, 20)
        ldbc_model = __pnoise_point_slopes__(fi, ldbc_fi, slopes, fm)
        func = intp.interp1d(log10(fm), ldbc_model, kind='linear')
        ldbc_0 = func(log10(fi[0]))
        ldbc_1 = func(log10(fi[1]))
        assert_almost_equal(ldbc_0, ldbc_fi[0], 0)
        assert_almost_equal(ldbc_1, ldbc_fi[1], 0)


    def test_with_points_slopes(self, plot=False):
        from copy import copy
        # test the new
        fi = np.array([1e4, 1e9])
        ldbc_fi = np.array([-40, -150])
        slopes = np.array([-30, -20])
        pnoise_model = Pnoise.with_points_slopes(fi, ldbc_fi, slopes)
        fm = np.logspace(3, 9, 20)
        pnoise_extrapolated = copy(pnoise_model)
        pnoise_extrapolated.fm = fm
        if plot:
            pnoise_model.plot('o')
            pnoise_extrapolated.plot('-x')
            plt.show()


    def test_integration(self, plot=False):
        fm = np.logspace(4,8,1000)
        lorentzian  = Pnoise(fm,10*np.log10(1/(fm*fm)),label='Lorentzian')
        white = Pnoise(fm,-120*np.ones(fm.shape),label='white')
        added = lorentzian+white
        iadded_gardner = added.integrate()
        iadded_trapz = added.integrate(method='trapz')
        f_int = lambda fm : 2.0e-12*fm-2.0/fm
        i_f_int = np.sqrt(f_int(fm[-1])-f_int(fm[0]))
        assert_almost_equal(iadded_trapz, i_f_int, 5)
        assert_almost_equal(iadded_trapz, i_f_int, 5)

    def test_add_points(self, plot=False):
        obj = Pnoise([1e3, 1e6, 1e9], [-100, -120, -140])
        obj.add_points([1e2, 1e3, 1e10], [-80, -70, -160])
        assert np.all(obj.fm == [1e2, 1e3, 1e6, 1e9, 1e10])
        assert np.all(obj.ldbc == [-80, -70, -120, -140, -160])
        # Check deep the interpolation
        obj.fm = [1e2, 1e6, 1.1e5, 7e7]
        obj.fm = [1e3, 1e6, 1.1e5,7e7, 1e10]


if __name__ == "__main__":
    unittest.main()
