from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter, uniform_filter, zoom
import matplotlib.pyplot as plt
import filter_array_recon_lib as far

show_mod_figures = False
show_fourier_figures = True
use_binning = True
apply_blurring = False
origin = ['green', 'red'][0] ## G at (0,0), or R at (0,0)
filename = ['Jumping spider03.jpg', 'Jumping spider05.jpg', 'autumn_tree.jpg', 'maroon_velvet.jpg',
         'emerald_satin.jpg', 'blue_velvet.jpg', 'spectrum_1024.png', 'spectrum.png',
         'PlatycryptusUndatusFemale.jpg'][8]
img = imread('./images/'+filename)[::-1,:,:]

## Ensure that the image has dimensions with an even number of pixels.
if use_binning:
    img = far.image_binning(img)
img = far.evencrop(img)
(Nx,Ny,_) = img.shape

plt.figure('original_img')
plt.imshow(array(img))

#zoombox = [880,1080,1430,1630]     ## PlatycryptusUndatusFemale.jpg
#zoombox = [580,720,1220,1440]     ## PlatycryptusUndatusFemale.jpg
zoombox = [880,1080,1280,1480]     ## PlatycryptusUndatusFemale.jpg
if use_binning:
    zoombox = array(zoombox) // 2
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

plt.figure('img_zoom')
plt.imshow(img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.axis('off')

if apply_blurring:
    img = far.image_blur(img, 'gaussian', 1, show_image=True)

(mm,nn) = indices((Nx,Ny))
mu_funcs = far.generate_bayer_modulation_functions(mm, nn, origin='green', show=show_mod_figures)
raw_img = far.generate_sampled_image_from_datacube(img, mu_funcs, zoom_region=zoombox, show=True)
fourier_recon = far.fourier_bayer_recon(raw_img, show=show_fourier_figures)
naive_recon = far.naive_bayer_recon(raw_img, origin=origin, upsample=True)

plt.figure('fourier_recon_img')
plt.imshow(fourier_recon)
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

plt.figure('fourier_recon_zoom')
plt.imshow(fourier_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.axis('off')

diff = far.augment_img_colordiff(img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:], fourier_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.figure('fourier_diff zoom')
plt.imshow(diff)
plt.axis('off')

plt.figure('naive_recon')
plt.imshow(naive_recon)

plt.figure('naive_recon_zoom')
plt.imshow(naive_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.axis('off')

naive_diff = far.augment_img_colordiff(img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:], naive_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:])
plt.figure('naive_diff zoom')
plt.imshow(naive_diff)
plt.axis('off')

plt.show()
