from numpy import *
import matplotlib.pyplot as plt
from imageio import imread
import matplotlib.gridspec as gridspec
import filter_array_recon_lib as far

window_function = ['rect','hanning','hamming','blackman','supergauss'][4]

filename = './images/buildings_and_sky_rgbpol.tif'
img = imread(filename)[::-1,:]      ## flip up-down to correct for origin at bottomleft
(Nx,Ny) = img.shape
print('(Nx,Ny)=', img.shape)

## Why does config #3 below work? Why not config #2, which seems to match Sony's website.
config = ['0-45-90-135', '90-45-0-135', '135-0-45-90'][2]

## Do both a naive reconstruction and a Fourier reconstruction.
(naive_s0,naive_ns1,naive_ns2) = far.naive_monopol_recon(img, config)
(fourier_s0,fourier_ns1,fourier_ns2) = far.fourier_monopol_recon(img, config, masktype=window_function, show=True)

## Show the naive reconstruction.
gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
fig1 = plt.figure('naive_recon', figsize=(18,6))

ax1 = fig1.add_subplot(gs[0])
ax1.set_title('normalized s1')
im1 = ax1.imshow(naive_ns1, vmin=-1, vmax=1, cmap='seismic')
ax1.axis('off')
ax1.axis('equal')

ax2 = fig1.add_subplot(gs[1])
ax2.set_title('normalized s2')
im2 = ax2.imshow(naive_ns2, vmin=-1, vmax=1, cmap='seismic')
ax2.axis('off')
ax2.axis('equal')

fig1.colorbar(im2, cax=fig1.add_subplot(gs[2]))

## Show the Fourier reconstruction.
gs = gridspec.GridSpec(1,3,width_ratios=[12,12,1])
fig2 = plt.figure('fourier_recon', figsize=(18,6))

ax1 = fig2.add_subplot(gs[0])
ax1.set_title('normalized s1')
im1 = ax1.imshow(fourier_ns1, vmin=-1, vmax=1, cmap='seismic')
ax1.axis('off')
ax1.axis('equal')

ax2 = fig2.add_subplot(gs[1])
ax2.set_title('normalized s2')
im2 = ax2.imshow(fourier_ns2, vmin=-1, vmax=1, cmap='seismic')
ax2.axis('off')
ax2.axis('equal')

fig2.colorbar(im2, cax=fig2.add_subplot(gs[2]))

plt.show()
