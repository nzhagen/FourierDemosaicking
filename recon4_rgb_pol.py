from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
import filter_array_recon_lib as far
import matplotlib.gridspec as gridspec

## Use image origin at bottom left.
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

def show_polarization_recon(label, ns1, ns2):
    ## Show the polarization reconstruction.
    if (ns1.ndim == 2):
        gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
        fig1 = plt.figure(label, figsize=(18,6))

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
        gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
        fig1 = plt.figure(label+'_R', figsize=(18,6))

        ax1 = fig1.add_subplot(gs[0])
        ax1.set_title(label+' normalized Rs1')
        im1 = ax1.imshow(ns1[:,:,0], vmin=-1, vmax=1, cmap='seismic')
        ax1.axis('off')
        ax1.axis('equal')

        ax2 = fig1.add_subplot(gs[1])
        ax2.set_title(label+' normalized Rs2')
        im2 = ax2.imshow(ns2[:,:,0], vmin=-1, vmax=1, cmap='seismic')
        ax2.axis('off')
        ax2.axis('equal')

        fig1.colorbar(im2, cax=fig1.add_subplot(gs[2]))

        gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
        fig2 = plt.figure(label+'_G', figsize=(18,6))

        ax1 = fig2.add_subplot(gs[0])
        ax1.set_title(label+' normalized Gs1')
        im1 = ax1.imshow(ns1[:,:,1], vmin=-1, vmax=1, cmap='seismic')
        ax1.axis('off')
        ax1.axis('equal')

        ax2 = fig2.add_subplot(gs[1])
        ax2.set_title(label+' normalized Gs2')
        im2 = ax2.imshow(ns2[:,:,1], vmin=-1, vmax=1, cmap='seismic')
        ax2.axis('off')
        ax2.axis('equal')

        fig2.colorbar(im2, cax=fig2.add_subplot(gs[2]))

        gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
        fig3 = plt.figure(label+'_B', figsize=(18,6))

        ax1 = fig3.add_subplot(gs[0])
        ax1.set_title(label+' normalized Bs1')
        im1 = ax1.imshow(ns1[:,:,2], vmin=-1, vmax=1, cmap='seismic')
        ax1.axis('off')
        ax1.axis('equal')

        ax2 = fig3.add_subplot(gs[1])
        ax2.set_title(label+' normalized Bs2')
        im2 = ax2.imshow(ns2[:,:,2], vmin=-1, vmax=1, cmap='seismic')
        ax2.axis('off')
        ax2.axis('equal')

        fig3.colorbar(im2, cax=fig3.add_subplot(gs[2]))

        avg_ns1 = mean(ns1, axis=2)
        avg_ns2 = mean(ns2, axis=2)

        alpha = 0.5 * rad2deg(arctan2(avg_ns2, avg_ns1))
        print(f'spatial average AOLP = {mean(alpha):.2f}deg')

        plt.figure(label+' AOLP')
        plt.imshow(alpha, vmin=-90, vmax=90, cmap='hsv')
        plt.colorbar()

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
        zoombox = [1276,1424,572,720]
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

    (rgb_s0, rgb_ns1, rgb_ns2) = far.fourier_rgbpol_recon(img, origin=origin, config=polconfig, masktype=window_function, show=False)
    show_polarization_recon('fourier', rgb_ns1, rgb_ns2)

    factor = 2**8 / 2**bit_depth    ## reduce bit-depth to fit into 8-bit display
    rgb_s0_uint8 = far.truncate_rgb_float_to_uint8(rgb_s0*factor)
    show_color_recon('fourier s0', rgb_s0_uint8, zoombox)

    (naive_rgb_s0_float, naive_rgb_ns1, naive_rgb_ns2) = far.naive_rgbpol_recon(img, origin=origin, config=polconfig, upsample=True)
    naive_rgb_s0 = far.truncate_rgb_float_to_uint8(naive_rgb_s0_float*factor)

    show_polarization_recon('naive', naive_rgb_ns1, naive_rgb_ns2)
    show_color_recon('naive s0', naive_rgb_s0, zoombox)

    plt.show()
