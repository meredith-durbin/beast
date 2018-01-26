#!/usr/bin/env python
"""
Plot the individual fit for a single observed star

.. history::
    Written 12 Jan 2016 by Karl D. Gordon
      based on code written by Heddy Arab for the BEAST techniques paper figure
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.patches import Rectangle
import matplotlib

from astropy.table import Table
from astropy.io import fits

from astropy import units as ap_units
from astropy.coordinates import SkyCoord as ap_SkyCoord

from .beastplotlib import initialize_parser

def add_rectangle(fig, ax1, ax2, label, 
                  linestyle='dashed', textorient='vertical',
                  xoffset=0.05, yoffset=0.04):
    '''
    Add bounding rectangles around set of subplots
    '''
    loc1 = ax1.get_position()
    loc2 = ax2.get_position()
    r = Rectangle(xy=(loc1.x0-xoffset, loc2.y0-yoffset),
               width=(loc2.x1 - loc1.x0)+(xoffset+0.02),
               height=(loc1.y1-loc2.y0)+(yoffset+0.01),
               transform=fig.transFigure,
               clip_on=False, zorder=-1,
               facecolor='none', edgecolor='k',
               linewidth=1, linestyle=linestyle)
    ax1.add_patch(r)
    if textorient == 'vertical':
        text_x = loc1.x0 - (xoffset+0.01)
        text_y = (loc1.y1 + loc1.y0 - yoffset)/2
    else:
        text_x = (loc1.x1 + loc1.x0 - xoffset)/2
        text_y = loc1.y1 + 0.02
    text = plt.figtext(x=text_x, y=text_y, s=label, rotation=textorient,
                       ha='center', va='center')

def setup_subplots(figsize=(8,8), usetex=True):
    matplotlib.rc('text', usetex=usetex)
    if usetex:
        matplotlib.rc('font', family='serif', size=12)
    fig = plt.figure(figsize=figsize)
    ax0 = plt.subplot2grid((5,4),(0,0),colspan=3,rowspan=3)

    ax1 = plt.subplot2grid((5,4),(3,0))
    ax2 = plt.subplot2grid((5,4),(3,1))
    ax3 = plt.subplot2grid((5,4),(3,2))

    ax4 = plt.subplot2grid((5,4),(4,0))
    ax5 = plt.subplot2grid((5,4),(4,1))
    ax6 = plt.subplot2grid((5,4),(4,2))

    ax7 = plt.subplot2grid((5,4),(3,3))
    ax8 = plt.subplot2grid((5,4),(4,3))
    
    axes = [ax1, ax2, ax3, ax7, ax4, ax5, ax6, ax8]

    for ax in axes:
        ax.set_ylim(0,1.1)
        ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.4, left=0.1)
    loc0 = ax0.get_position()
    loc0.y0 += 0.03
    ax0.set_position(loc0)

    add_rectangle(fig, ax1, ax3, '\em{Primary}')
    add_rectangle(fig, ax4, ax6, '\em{Secondary}', linestyle='dotted')
    add_rectangle(fig, ax7, ax8, '\em{Derived}', linestyle='dashdot',
                  xoffset=0.02, textorient='horizontal')

    ax1.set_ylabel('Probability')
    ax4.set_ylabel('Probability')
    
    return fig, axes + [ax0]

def disp_str(stats, k, keyname):
    dvals = [stats[keyname+'_p50'][k],
             stats[keyname+'_p84'][k],
             stats[keyname+'_p16'][k]]
    if keyname == 'M_ini':
        dvals = np.log10(dvals)
    disp_str = '$' + \
               "{0:.2f}".format(dvals[0]) + \
               '^{+' + \
               "{0:.2f}".format(dvals[1] - dvals[0]) + \
               '}_{-' + \
               "{0:.2f}".format(dvals[0] - dvals[2]) + \
               '}$'

    return disp_str


def plot_1dpdf(ax, pdf1d_hdu, tagname, xlabel, starnum,
               stats=None, logx=False):

    pdf = pdf1d_hdu[tagname].data

    # n_objects, n_bins = pdf.shape
    # n_objects -= 1

    xvals = pdf[-1,:]
    if logx:
        xvals = np.log10(xvals)

    # if tagname == 'Z':
    #     gindxs, = np.where(pdf[starnum,:] > 0.)
    #     pdf_scaled = pdf[starnum,gindxs]/max(pdf[starnum,gindxs])
    #     print(pdf_scaled)
    #     ax.plot(xvals[gindxs], pdf_scaled, color='k')
    # else:
    #gindxs = (pdf[starnum,:] > 0.)
    pdf_scaled = pdf[starnum,:]/max(pdf[starnum,:])
    ax.plot(xvals,pdf_scaled,color='k')

    ax.xaxis.set_major_locator(MaxNLocator(3))
    xlim = [xvals.min(), xvals.max()]
    xlim_delta = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0]-0.05*xlim_delta, xlim[1]+0.05*xlim_delta)
    pdfmax = pdf_scaled.argmax()
    if (xvals[pdfmax] - xlim[0])/xlim_delta >= 0.5:
        text_x, text_ha = 0.05, 'left'
    else:
        text_x, text_ha = 0.95, 'right'
    ax.text(text_x, 0.95, xlabel, transform=ax.transAxes,
            va='top', ha=text_ha)

    if stats is not None:
        ylim = ax.get_ylim()

        y1 = ylim[0] + 0.5*(ylim[1]-ylim[0])
        y2 = ylim[0] + 0.7*(ylim[1]-ylim[0])
        pval = stats[tagname+'_Best'][starnum]
        if logx:
            pval = np.log10(pval)
        ax.plot(np.full((2),pval),[y1,y2],
                '-', color='c')

        y1 = ylim[0] + 0.2*(ylim[1]-ylim[0])
        y2 = ylim[0] + 0.4*(ylim[1]-ylim[0])
        y1m = ylim[0] + 0.25*(ylim[1]-ylim[0])
        y2m = ylim[0] + 0.35*(ylim[1]-ylim[0])
        ym = 0.5*(y1 + y2)
        pvals = [stats[tagname+'_p50'][starnum],
                 stats[tagname+'_p16'][starnum],
                 stats[tagname+'_p84'][starnum]]
        if logx:
            pvals = np.log10(pvals)
        ax.plot(np.full((2),pvals[0]),[y1m,y2m],'-', color='m')
        ax.plot(np.full((2),pvals[1]),[y1,y2],'-', color='m')
        ax.plot(np.full((2),pvals[2]),[y1,y2],'-', color='m')
        ax.plot(pvals[1:3],[ym,ym],'-', color='m')

def plot_beast_ifit(filters, waves, stats, pdf1d_hdu, starnum):
    k = starnum

    fig, ax = setup_subplots()

    n_filters = len(filters)
    
    # get the observations
    waves *= 1e-4
    obs_flux = np.empty((n_filters),dtype=np.float)
    mod_flux = np.empty((n_filters,3),dtype=np.float)
    mod_flux_nd = np.empty((n_filters,3),dtype=np.float)
    mod_flux_wbias = np.empty((n_filters,3),dtype=np.float)

    c = ap_SkyCoord(ra=stats['RA'][k]*ap_units.degree,
                    dec=stats['DEC'][k]*ap_units.degree,
                    frame='icrs')
    corname = ('PHAT J' + 
               c.ra.to_string(unit=ap_units.hourangle, sep="",precision=2,
                              alwayssign=False,pad=True) + 
               c.dec.to_string(sep="",precision=2,
                               alwayssign=True,pad=True))
        
    for i, cfilter in enumerate(filters):
        obs_flux[i] = stats[cfilter][k]
        mod_flux[i,0] = np.power(10.0,stats['log'+cfilter+'_wd_p50'][k])
        mod_flux[i,1] = np.power(10.0,stats['log'+cfilter+'_wd_p16'][k])
        mod_flux[i,2] = np.power(10.0,stats['log'+cfilter+'_wd_p84'][k])
        mod_flux_nd[i,0] = np.power(10.0,stats['log'+cfilter+'_nd_p50'][k])
        mod_flux_nd[i,1] = np.power(10.0,stats['log'+cfilter+'_nd_p16'][k])
        mod_flux_nd[i,2] = np.power(10.0,stats['log'+cfilter+'_nd_p84'][k])
        if 'log'+cfilter+'_wd_bias_p50' in stats.colnames:
            mod_flux_wbias[i,0] = np.power(10.0,stats['log'+cfilter+
                                                      '_wd_bias_p50'][k])
            mod_flux_wbias[i,1] = np.power(10.0,stats['log'+cfilter+
                                                      '_wd_bias_p16'][k])
            mod_flux_wbias[i,2] = np.power(10.0,stats['log'+cfilter+
                                                      '_wd_bias_p84'][k])
        
    ax[8].plot(waves, obs_flux, 'ko', label='observed')

    if 'log'+filters[0]+'_wd_bias_p50' in stats.colnames:
        ax[8].plot(waves, mod_flux_wbias[:,0], 'b-',label='stellar+dust+bias')
        ax[8].fill_between(waves, mod_flux_wbias[:,1], mod_flux_wbias[:,2],
                           color='b', alpha = 0.3)

    ax[8].plot(waves, mod_flux[:,0], 'r-',label='stellar+dust')
    ax[8].fill_between(waves, mod_flux[:,1], mod_flux[:,2],
                       color='r', alpha = 0.2)

    ax[8].plot(waves, mod_flux_nd[:,0], 'y-',label='stellar only')
    ax[8].fill_between(waves, mod_flux_nd[:,1], mod_flux_nd[:,2],
                       color='y', alpha = 0.1)

    ax[8].legend(loc='upper right', bbox_to_anchor=(1.4, 1.025))

    ax[8].set_ylabel(r'Flux [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
    ax[8].set_yscale('log')

    ax[8].set_xscale('log', minor=False)
    ax[8].set_xlabel(r'$\lambda$ [$\mu$m]')
    ax[8].set_xlim(0.2,2.0)
    ax[8].minorticks_off()
    ax[8].set_xticks([0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0])
    ax[8].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax[8].text(0.95, 0.95, corname, transform=ax[8].transAxes,
               va='top',ha='right')

    # add the text results
    keys = ['Av','M_ini','logA','Rv','f_A','Z','logT','logg','logL']
    dispnames = ['A(V)','log(M)','log(t)','R(V)',r'f$_\mathcal{A}$','Z',
                 r'log(T$_\mathrm{eff})$','log(g)','log(L)']
    laby = 0.7
    ty = np.linspace(laby-0.07,0.1,num=len(keys))
    ty[3:] -= 0.035
    ty[6:] -= 0.035
    tx = [1.14, 1.22, 1.34]
    for i in range(len(keys)):
        ax[8].text(tx[0], ty[i], dispnames[i],
                   ha='right',
                   transform=ax[8].transAxes)
        ax[8].text(tx[1], ty[i], disp_str(stats, starnum, keys[i]),
                   ha='center', color='m',
                   transform=ax[8].transAxes)
        best_val = stats[keys[i]+'_Best'][k]
        if keys[i] == 'M_ini':
            best_val = np.log10(best_val)
        ax[8].text(tx[2], ty[i],
                   '$' + "{0:.2f}".format(best_val) + '$', 
                   ha='center', color='c', 
                   transform=ax[8].transAxes)
    ax[8].text(tx[0],laby, 'Param',
               ha='right',
               transform=ax[8].transAxes)
    ax[8].text(tx[1],laby, r'50\%$\pm$33%',
               ha='center', color='k',
               transform=ax[8].transAxes)
    ax[8].text(tx[2],laby, 'Best',color='k',
               ha='center',
               transform=ax[8].transAxes)

    # now draw boxes around the different kinds of parameters
    tax = ax[8]

    # primary
    rec = Rectangle((tx[0]-0.13,ty[2]-0.02),
                    tx[2]-tx[0]+0.17, (ty[0]-ty[2])*1.6,
                    fill=False, lw=1, transform=tax.transAxes,
                    ls='dashed', clip_on=False)
    tax.add_patch(rec)

    # secondary
    rec = Rectangle((tx[0]-0.13,ty[5]-0.02),
                    tx[2]-tx[0]+0.17, (ty[0]-ty[2])*1.6,
                    fill=False, lw=1, transform=tax.transAxes,
                    ls='dotted', clip_on=False)
    tax.add_patch(rec)

    # derived
    rec = Rectangle((tx[0]-0.13,ty[8]-0.02),
                    tx[2]-tx[0]+0.17, (ty[0]-ty[2])*1.6,
                    fill=False, lw=1, transform=tax.transAxes,
                    ls='dashdot', clip_on=False)
    tax.add_patch(rec)

    # plot the primary parameter 1D PDFs
    plot_1dpdf(ax[0], pdf1d_hdu, 'Av', 'A(V)', starnum,
               stats=stats)
    plot_1dpdf(ax[1], pdf1d_hdu, 'M_ini', 'log(M)', starnum, logx=True,
               stats=stats)
    plot_1dpdf(ax[2], pdf1d_hdu, 'logA', 'log(t)', starnum,
               stats=stats)

    # plot the secondary parameter 1D PDFs
    plot_1dpdf(ax[4], pdf1d_hdu, 'Rv', 'R(V)', starnum,
               stats=stats)
    plot_1dpdf(ax[5], pdf1d_hdu, 'f_A', r'f$_\mathcal{A}$', starnum,
               stats=stats)
    plot_1dpdf(ax[6], pdf1d_hdu, 'Z', 'Z', starnum,
               stats=stats)


    # plot the derived parameter 1D PDFs
    plot_1dpdf(ax[3], pdf1d_hdu, 'logT', r'log(T$_\mathrm{eff})$', starnum,
               stats=stats)
    plot_1dpdf(ax[7], pdf1d_hdu, 'logg', 'log(g)', starnum,
               stats=stats)


if __name__ == '__main__':

    parser = initialize_parser()
    parser.add_argument("filebase", type=str,
                        help='base filename of output results')
    parser.add_argument("--starnum", type=int, default=0,
                        help="star number in observed file")
    args = parser.parse_args()

    starnum = args.starnum

    # base filename
    filebase = args.filebase

    # read in the stats
    stats = Table.read(filebase + '_stats.fits')

    # open 1D PDF file
    pdf1d_hdu = fits.open(filebase+'_pdf1d.fits')

    # filters for PHAT
    filters = ['HST_WFC3_F275W','HST_WFC3_F336W','HST_ACS_WFC_F475W',
               'HST_ACS_WFC_F814W','HST_WFC3_F110W','HST_WFC3_F160W']
    waves = np.asarray([2722.05531502, 3366.00507206,4763.04670013,
                        8087.36760191,11672.35909295,15432.7387546])
    
    fig, ax = plt.subplots(figsize=(8,8))

    # make the plot!
    plot_beast_ifit(filters, waves, stats, pdf1d_hdu, starnum)

    # show or save
    basename = filebase + '_ifit_starnum_' + str(starnum)
    print(basename)
    if args.savefig:
        fig.savefig('{}.{}'.format(basename, args.savefig))
    else:
        plt.show()

    
