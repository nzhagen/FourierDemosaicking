from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
import filter_array_recon_lib as far
import matplotlib.gridspec as gridspec

## Use image origin at bottom left.
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

def show_polarization_recon(label, s0, ns1, ns2):
    ## Show the polarization reconstruction.
    if (ns1.ndim == 2):
        gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
        fig1 = plt.figure(label, figsize=(9,3)) #figsize=(18,6))

        ax1 = fig1.add_subplot(gs[0])
        ax1.set_title(label+' normalized s1')
        im1 = ax1.imshow(ns1, vmin=-1, vmax=1, cmap='seismic')
        ax1.axis('off')
        ax1.axis('equal')

        ax2 = fig1.add_subplot(gs[1])
        ax2.set_title(label+' normalized s2')
        im2 = ax2.imshow(ns2, vmin=-1, vmax=1, cmap='seismic')
        ax2.axis('off')
        ax2.axis('equal')

        fig1.colorbar(im2, cax=fig1.add_subplot(gs[2]))

        alpha = 0.5 * rad2deg(arctan2(avg_ns2, avg_ns1))
        print(f'spatial average AOLP = {mean(alpha):.2f}deg')

        plt.figure(label+' AOLP')
        plt.imshow(alpha, vmin=-90, vmax=90, cmap='hsv')
        plt.colorbar()
    elif (ns1.ndim == 3):
        colors = ['R','G','B']

        s0max = amax(s0)
        for i,c in enumerate(colors):
            figname = label+'_'+c+'_s0'
            plt.figure(figname, figsize=(4,3))
            plt.title(figname)
            plt.imshow(s0[:,:,i], vmin=0, vmax=s0max, cmap='gray')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(figname+'.pdf')

            figname = label+'_'+c+'_ns1'
            plt.figure(figname, figsize=(4,3))
            plt.title(figname)
            plt.imshow(ns1[:,:,i], vmin=-1, vmax=1, cmap='seismic')
            plt.axis('off')
            cbar = plt.colorbar()
            cbar.set_ticks([-1.0,-0.5,0,0.5,1.0])
            plt.savefig(figname+'.pdf')

            figname = label+'_'+c+'_ns2'
            plt.figure(figname, figsize=(4,3))
            plt.title(figname)
            plt.imshow(ns2[:,:,i], vmin=-1, vmax=1, cmap='seismic')
            plt.axis('off')
            cbar = plt.colorbar()
            cbar.set_ticks([-1.0,-0.5,0,0.5,1.0])
            plt.savefig(figname+'.pdf')

            alpha = 0.5 * rad2deg(arctan2(ns2[:,:,i], ns1[:,:,i]))
            figname = label+'_'+c+'_AOLP'
            plt.figure(figname, figsize=(4,3))
            plt.title(figname)
            plt.imshow(alpha, vmin=-90, vmax=90, cmap='hsv')
            plt.axis('off')
            cbar = plt.colorbar()
            cbar.set_ticks([-90,-45,0,45,90])
            plt.savefig(figname+'.pdf')

        avg_ns1 = mean(ns1, axis=2)
        avg_ns2 = mean(ns2, axis=2)

        alpha = 0.5 * rad2deg(arctan2(avg_ns2, avg_ns1))
        print(f'spatial average AOLP = {mean(alpha):.2f}deg')

        plt.figure(label+' avg AOLP')
        plt.imshow(alpha, vmin=-90, vmax=90, cmap='hsv')
        cbar = plt.colorbar()
        cbar.set_ticks([-90,-45,0,45,90])
        plt.savefig(label+'_avg_AOLP.pdf')

    return

def show_color_recon(figname, img, zoombox=None):
    plt.figure(figname)
    plt.imshow(img)

    if zoombox is not None:
        plt.figure(figname)
        plt.plot(hbox, vbox, 'g-', lw=3)

        zoom = img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]

        plt.figure(figname+'_zoom')
        plt.imshow(zoom)
        plt.axis('off')

    return

## =================================================================================================
## =================================================================================================

if (__name__ == '__main__'):
    simulate = False
    polconfig = '135-0-45-90'
    window_function = ['rect','hanning','hamming','blackman','supergauss'][1]

    if simulate:
        blurring = 1        ## "1" means no blurring
        #blurring = 4
        binning = 1
        origin = ['G','R'][0]    ## G at (0,0) or R at (0,0)
        bit_depth = 8
        imgpol = '22.5'    ## angle of polarization of image
        filename = ['autumn_tree.jpg', 'spectrum.png', 'PlatycryptusUndatusFemale.jpg', 'Phidippus_regius.jpg'][2]
        (dcb,img) = far.simulate_rgbpol_rawimg_from_dcb(filename, imgpol, origin, binning, blurring, polconfig=polconfig, show=False)
        zoombox = [880,1080,1280,1480]     ## (xlo,xhi,ylo,yhi] PlatycryptusUndatusFemale.jpg
        if (binning > 1):
            zoombox = array(zoombox) // binning
    else:
        img = imread('/home/nh/repos/FourierDemosaicking/images/roadway_rgbpol.tif')
        bit_depth = 13
        #zoombox = [1276,1424,572,720]
        zoombox = [1276-200,1424+200,572-200,720+200]
        origin = 'G'

    if zoombox:
        ## Define box coordinates to draw on the image, to show the zoom region.
        hbox = [zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]]
        vbox = [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]]

    plt.figure('input raw image')
    plt.imshow(img)
    plt.colorbar()
    if zoombox:
        plt.plot(hbox, vbox, 'g-', lw=3)

    (rgb_s0, rgb_ns1, rgb_ns2) = far.fourier_rgbpol_recon(img, origin=origin, config=polconfig, masktype=window_function, show=True)
    #show_polarization_recon('fourier', rgb_s0, rgb_ns1, rgb_ns2)

    if zoombox:
        fourier_s0_zoom = rgb_s0[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        fourier_ns1_zoom = rgb_ns1[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        fourier_ns2_zoom = rgb_ns2[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        show_polarization_recon('fourier', fourier_s0_zoom, fourier_ns1_zoom, fourier_ns2_zoom)

    factor = 2**8 / 2**bit_depth    ## reduce bit-depth to fit into 8-bit display
    rgb_s0_uint8 = far.truncate_rgb_float_to_uint8(rgb_s0*factor)
    show_color_recon('fourier s0', rgb_s0_uint8, zoombox)

    (naive_rgb_s0, naive_rgb_ns1, naive_rgb_ns2) = far.naive_rgbpol_recon(img, origin=origin, config=polconfig, upsample=True)
    #naive_rgb_s0_uint8 = far.truncate_rgb_float_to_uint8(naive_rgb_s0*factor)

    #show_polarization_recon('naive', naive_rgb_s0, naive_rgb_ns1, naive_rgb_ns2)
    show_color_recon('naive s0', naive_rgb_s0, zoombox)

    if zoombox:
        naive_s0_zoom = naive_rgb_s0[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        naive_ns1_zoom = naive_rgb_ns1[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        naive_ns2_zoom = naive_rgb_ns2[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        show_polarization_recon('naive', naive_s0_zoom, naive_ns1_zoom, naive_ns2_zoom)

    plt.show()
