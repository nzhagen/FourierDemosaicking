from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread, imsave
import filter_array_recon_lib as far

## ============================================================
## ============================================================

show_figures = True
blurring = 0        ## "0" means no blurring
#blurring = 4
binning = 1
origin = ['G','R'][0]    ## G at (0,0) or R at (0,0)
window_function = ['rect','hanning','hamming','blackman','supergauss'][4]
simulate = True

if simulate:
    filename = ['autumn_tree.jpg', 'spectrum.png', 'PlatycryptusUndatusFemale.jpg'][0]
    zoombox = [880,1080,1280,1480]     ## (xlo,xhi,ylo,yhi] PlatycryptusUndatusFemale.jpg
    if (binning > 1):
        zoombox = array(zoombox) // binning
    (dcb,raw_img) = far.simulate_quadbayer_rawimg_from_dcb(filename, origin, binning, blurring, show=show_figures)
else:
    raw_img = imread('./images/roadway_rgbpol.tif') // 16       ## divide by 16 to convert 12-bit to 8-bit
    dcb = False
    zoombox = [1272,1422,577,727]

fourier_recon_float = far.fourier_quadbayer_recon(raw_img, origin, masktype=window_function, show=show_figures)
fourier_recon = far.truncate_rgb_float_to_uint8(fourier_recon_float)
naive_recon_float = far.naive_quadbayer_recon(raw_img, origin, upsample=True)
naive_recon = far.truncate_rgb_float_to_uint8(naive_recon_float)

if zoombox:
    ## Define box coordinates to draw on the image, to show the zoom region.
    hbox = [zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]]
    vbox = [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]]

if any(dcb):
    plt.figure('original_dcb')
    plt.imshow(dcb)
    if zoombox:
        plt.plot(hbox, vbox, '-', color=[0,1,0], lw=3)

plt.figure('fourier_recon_img')
plt.imshow(fourier_recon)
if zoombox:
    plt.plot(hbox, vbox, '-', color=[0,1,0], lw=3)

    fourier_recon_zoom = fourier_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
    naive_recon_zoom = naive_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]

    plt.figure('fourier_recon_zoom')
    plt.imshow(fourier_recon_zoom)
    plt.axis('off')

    if any(dcb):
        dcb_zoom = dcb[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
        fourier_diff_quant = sum(float32(fourier_recon_zoom) - float32(dcb_zoom), axis=2)
        plt.figure('fourier_diff_quant_zoom')
        plt.imshow(fourier_diff_quant)
        plt.axis('off')
        plt.colorbar()

plt.figure('naive_recon')
plt.imshow(naive_recon)

if zoombox:
    plt.figure('naive_recon_zoom')
    plt.imshow(naive_recon_zoom)
    plt.axis('off')

    if any(dcb):
        naive_diff_quant = sum(float32(naive_recon_zoom) - float32(dcb_zoom), axis=2)
        plt.figure('naive_diff_quant_zoom')
        plt.imshow(naive_diff_quant)
        plt.axis('off')
        plt.colorbar()

plt.show()

