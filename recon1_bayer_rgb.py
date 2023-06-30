from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
import matplotlib.pyplot as plt
import filter_array_recon_lib as far

## Use image origin at bottom left.
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

show_mod_figures = False
show_fourier_figures = True
binning = 1     ## '1' means no binning
blurring = 1    ## '1' means no blurring
origin = ['G', 'R'][0] ## G at (0,0), or R at (0,0)
filename = ['autumn_tree.jpg', 'spectrum.png', 'PlatycryptusUndatusFemale.jpg'][2]
img = imread('./images/'+filename)[::-1,:,:]        ## flip up-down to correct for origin at bottomleft

## Ensure that the image has dimensions with an even number of pixels.
if (binning > 1):
    img = far.image_binning(img)
img = far.evencrop(img)
(Nx,Ny,_) = img.shape

plt.figure('original_img')
plt.imshow(array(img))

#zoombox = [880,1080,1430,1630]     ## PlatycryptusUndatusFemale.jpg
#zoombox = [580,720,1220,1440]     ## PlatycryptusUndatusFemale.jpg
zoombox = [880,1080,1280,1480]     ## PlatycryptusUndatusFemale.jpg
if (binning > 1):
    zoombox = array(zoombox) // 2
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

plt.figure('img_zoom')
plt.imshow(img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.axis('off')

mu_funcs = far.generate_bayer_modulation_functions(Nx, Ny, origin, show=show_mod_figures)
raw_img = far.generate_sampled_image_from_datacube(img, mu_funcs, zoom_region=zoombox, show=True)
fourier_recon_float = far.fourier_bayer_recon(raw_img, show=show_fourier_figures)
fourier_recon = far.truncate_rgb_float_to_uint8(fourier_recon_float)
naive_recon_float = far.naive_bayer_recon(raw_img, origin=origin, upsample=True)
naive_recon = far.truncate_rgb_float_to_uint8(naive_recon_float)

img_zoom = img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
fourier_recon_zoom = fourier_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
naive_recon_zoom = naive_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]

plt.figure('fourier_recon_img')
plt.imshow(fourier_recon)
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

plt.figure('fourier_recon_zoom')
plt.imshow(fourier_recon_zoom)
plt.axis('off')

#diff = far.augment_img_colordiff(img_zoom, fourier_recon_zoom)
#plt.figure('fourier_diff zoom')
#plt.imshow(diff)
#plt.axis('off')

fourier_diff_quant = sum(float32(fourier_recon_zoom) - float32(img_zoom), axis=2)
plt.figure('fourier_diff_quant_zoom')
plt.imshow(fourier_diff_quant)
plt.axis('off')
plt.colorbar()

plt.figure('naive_recon')
plt.imshow(naive_recon)

plt.figure('naive_recon_zoom')
plt.imshow(naive_recon_zoom)
plt.axis('off')

#naive_diff = far.augment_img_colordiff(img_zoom, naive_recon_zoom)
#plt.figure('naive_diff zoom')
#plt.imshow(naive_diff)
#plt.axis('off')

naive_diff_quant = sum(float32(naive_recon_zoom) - float32(img_zoom), axis=2)
plt.figure('naive_diff_quant_zoom')
plt.imshow(naive_diff_quant)
plt.axis('off')
plt.colorbar()

plt.show()

