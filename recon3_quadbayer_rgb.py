from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
import filter_array_recon_lib as far

## ============================================================
## ============================================================

show_mu_figures = False
show_fourier_figures = True
blurring = 1        ## "1" means no blurring
#blurring = 4
use_binning = False
origin = ['R','G'][1]    ## R at (0,0) or G at (0,0)

filename = ['autumn_tree.jpg', 'spectrum.png', 'PlatycryptusUndatusFemale.jpg'][2]
img = imread('./images/'+filename)[::-1,:,:]
#img = image_binning(img)
img = far.evencrop(img)

#zoombox = [880,1080,1430,1630]     ## PlatycryptusUndatusFemale.jpg
#zoombox = [580,720,1220,1440]     ## PlatycryptusUndatusFemale.jpg
zoombox = [880,1080,1280,1480]     ## PlatycryptusUndatusFemale.jpg
if use_binning:
    zoombox = array(zoombox) // 2

(Nx,Ny,_) = img.shape
(Px,Py) = (Nx//2, Ny//2)
(Mx,My) = (Px//2, Py//2)  ## mask size
print(f'(Nx,Ny)=({Nx},{Ny}), (Px,Py)=({Px},{Py}), (Mx,My)=({Mx},{My})')

plt.figure('original_img')
plt.imshow(array(img))

if blurring > 1:
    img = far.image_blur(img, 'gaussian', blurring, show_image=True)

(mm,nn) = indices((Nx,Ny))
(mu_r, mu_g, mu_b) = far.generate_quadbayer_modulation_functions(mm, nn, origin='G', show=show_mu_figures)

meas_img = (img[:,:,0] * mu_r) + (img[:,:,1] * mu_g) + (img[:,:,2] * mu_b)
meas_img_rgb = zeros_like(img)
meas_img_rgb[:,:,0] = img[:,:,0] * mu_r
meas_img_rgb[:,:,1] = img[:,:,1] * mu_g
meas_img_rgb[:,:,2] = img[:,:,2] * mu_b

plt.figure('meas_img')
plt.imshow(meas_img_rgb)

fourier_recon = far.fourier_quadbayer_recon(meas_img, origin, show=show_fourier_figures)
naive_recon = far.naive_quadbayer_recon(meas_img, origin, upsample=True)

plt.figure('original_img')
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

plt.figure('fourier_recon_img')
plt.imshow(fourier_recon)
plt.plot([zoombox[2],zoombox[3],zoombox[3],zoombox[2],zoombox[2]], [zoombox[0],zoombox[0],zoombox[1],zoombox[1],zoombox[0]], '-', color=[0,1,0], lw=3)

img_zoom = img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
fourier_recon_zoom = fourier_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]
naive_recon_zoom = naive_recon[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3],:]

plt.figure('fourier_recon_zoom')
plt.imshow(fourier_recon_zoom)
plt.axis('off')

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

naive_diff_quant = sum(float32(naive_recon_zoom) - float32(img_zoom), axis=2)
plt.figure('naive_diff_quant_zoom')
plt.imshow(naive_diff_quant)
plt.axis('off')
plt.colorbar()

plt.show()

