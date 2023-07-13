from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.transform import resize
import matplotlib.pyplot as plt
import struct

## Use image origin at bottom left.
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

## ============================================================
def iseven(x):
    return (x % 2 == 0)

## ============================================================
def isodd(x):
    return (x % 2 != 0)

## ===============================================================================================
def is_number(s):
    ## Check if a string represents a number --- any number.
    try:
        float(s)
        return True
    except ValueError:
        return False

## ============================================================
def image_binning(img):
    ## We average (rather than sum) the binned output in order to keep to 8-bit data type.
    output = float32(img[0::2,0::2]) + float32(img[1::2,0::2]) + float32(img[0::2,1::2]) + float32(img[1::2,1::2])
    output = uint8(output / 4.0)
    return(output)

## ============================================================
def image_blur(img, filter_type='gaussian', filter_size=2, show_image=False):
    ## Apply blurring to the image. The larger the "filter_size" value, the more the blurring.

    if (filter_type == 'gaussian'):
        output = array(img)
        output[:,:,0] = gaussian_filter(img[:,:,0], sigma=filter_size, mode='wrap')
        output[:,:,1] = gaussian_filter(img[:,:,1], sigma=filter_size, mode='wrap')
        output[:,:,2] = gaussian_filter(img[:,:,2], sigma=filter_size, mode='wrap')
        output = uint8(output)
    elif (filter_type == 'uniform'):
        output = array(img)
        output[:,:,0] = uniform_filter(img[:,:,0], size=filter_size, mode='wrap')
        output[:,:,1] = uniform_filter(img[:,:,1], size=filter_size, mode='wrap')
        output[:,:,2] = uniform_filter(img[:,:,2], size=filter_size, mode='wrap')
        output = uint8(output)

    if show_image:
        plt.figure('blurred_img')
        plt.imshow(output)
        plt.colorbar()

    return(output)

## ============================================================
def draw_bayer_fft_circles(Nx,Ny, normalize=False):
    ## Draw circles (or ellipses, if Nx != Ny).

    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)

    if normalize:
        radius = 0.5
        cen1 = ( 0, 0)
        cen2 = ( 1, 0)
        cen3 = (-1, 0)
        cen4 = ( 0,-1)
        cen5 = ( 0, 1)
        cen6 = ( 1, 1)
        cen7 = (-1,-1)
        cen8 = (-1, 1)
        cen9 = ( 1,-1)
    else:
        radius = Px-Mx
        cen1 = ((Nx-1)/2, (Ny-1)/2)
        cen2 = (Nx-1, (Ny-1)/2)
        cen3 = (0.0, (Ny-1)/2)
        cen4 = ((Nx-1)/2, 0.0)
        cen5 = ((Nx-1)/2, (Ny-1))
        cen6 = ((Nx-1), (Ny-1))
        cen7 = (0.0, 0.0)
        cen8 = (0.0, (Ny-1))
        cen9 = ((Nx-1), 0.0)

    thetas = linspace(0.0, 2.0*pi, 200)
    x1 = cen1[1] + radius * cos(thetas)
    y1 = cen1[0] + radius * sin(thetas)

    thetas = linspace(pi, 2.0*pi, 100)
    x2 = cen2[1] + radius * cos(thetas)
    y2 = cen2[0] + radius * sin(thetas)

    thetas = linspace(0.0, pi, 100)
    x3 = cen3[1] + radius * cos(thetas)
    y3 = cen3[0] + radius * sin(thetas)

    thetas = linspace(3.0*pi/2.0, 5.0*pi/2.0, 100)
    x4 = cen4[1] + radius * cos(thetas)
    y4 = cen4[0] + radius * sin(thetas)

    thetas = linspace(pi/2.0, 3.0*pi/2.0, 100)
    x5 = cen5[1] + radius * cos(thetas)
    y5 = cen5[0] + radius * sin(thetas)

    thetas = linspace(pi, 3.0*pi/2.0, 100)
    x6 = cen6[1] + radius * cos(thetas)
    y6 = cen6[0] + radius * sin(thetas)

    thetas = linspace(0.0, pi/2.0, 100)
    x7 = cen7[1] + radius * cos(thetas)
    y7 = cen7[0] + radius * sin(thetas)

    thetas = linspace(pi/2.0, pi, 100)
    x8 = cen8[1] + radius * cos(thetas)
    y8 = cen8[0] + radius * sin(thetas)

    thetas = linspace(3.0*pi/2.0, 2.0*pi, 100)
    x9 = cen9[1] + radius * cos(thetas)
    y9 = cen9[0] + radius * sin(thetas)

    plt.plot(x1, y1, 'k-', lw=3)
    plt.plot(x2, y2, 'b-', lw=3)
    plt.plot(x3, y3, 'b-', lw=3)
    plt.plot(x4, y4, 'r-', lw=3)
    plt.plot(x5, y5, 'r-', lw=3)
    plt.plot(x6, y6, 'g-', lw=3)
    plt.plot(x7, y7, 'g-', lw=3)
    plt.plot(x8, y8, 'g-', lw=3)
    plt.plot(x9, y9, 'g-', lw=3)

    return

## ============================================================
def draw_monopol_fft_circles(Nx,Ny, normalize=True, alpha=1.0):
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)

    if normalize:
        radius = 0.5
        cen1 = ( 0, 0)
        cen2 = ( 1, 0)
        cen3 = (-1, 0)
        cen4 = ( 0,-1)
        cen5 = ( 0, 1)
    else:
        radius = Px-Mx
        cen1 = ((Nx-1)/2.0, (Ny-1)/2.0)
        cen2 = (Nx-1, (Ny-1)/2.0)
        cen3 = (0.0, (Ny-1)/2.0)
        cen4 = ((Nx-1)/2.0, 0.0)
        cen5 = ((Nx-1)/2.0, (Ny-1))

    thetas = linspace(0.0, 2.0*pi, 200)
    x1 = cen1[1] + radius * cos(thetas)
    y1 = cen1[0] + radius * sin(thetas)

    thetas = linspace(pi, 2.0*pi, 100)
    x2 = cen2[1] + radius * cos(thetas)
    y2 = cen2[0] + radius * sin(thetas)

    thetas = linspace(0.0, pi, 100)
    x3 = cen3[1] + radius * cos(thetas)
    y3 = cen3[0] + radius * sin(thetas)

    thetas = linspace(3.0*pi/2.0, 5.0*pi/2.0, 100)
    x4 = cen4[1] + radius * cos(thetas)
    y4 = cen4[0] + radius * sin(thetas)

    thetas = linspace(pi/2.0, 3.0*pi/2.0, 100)
    x5 = cen5[1] + radius * cos(thetas)
    y5 = cen5[0] + radius * sin(thetas)

    plt.plot(x1, y1, 'k-', lw=3, alpha=alpha)
    plt.plot(x2, y2, 'b-', lw=3, alpha=alpha)
    plt.plot(x3, y3, 'b-', lw=3, alpha=alpha)
    plt.plot(x4, y4, 'r-', lw=3, alpha=alpha)
    plt.plot(x5, y5, 'r-', lw=3, alpha=alpha)

    return

## ============================================================
def closest_n_divisible_by_m(n, m):
    ## Find the number closest to n and divisible by m.
    q = int(n / m)

    ## n1 = first possible closest number, n2 = secon possible number. Then check which one to use.
    n1 = m * q

    if((n * m) > 0) :
        n2 = (m * (q + 1))
    else :
        n2 = (m * (q - 1))

    ## If the statement below is true, then n1 is the number we want.
    if (abs(n - n1) < abs(n - n2)) :
        res = n1
    else:
        res = n2

    ## If the result is larger than n, then subtract m to get the nearest lower value.
    if (res > n):
        res -= m

    return(res)

## ============================================================
def evencrop(input_img, verbose=False):
    ## Crop pixels from an image until its dimensions reach an even factor of 2 (a factor of 8 to be precise).

    img = array(input_img)
    (Nx,Ny,_) = img.shape

    new_Nx = closest_n_divisible_by_m(Nx, 8)
    new_Ny = closest_n_divisible_by_m(Ny, 8)

    if (new_Nx != Nx):
        img = img[0:new_Nx,:,:]
    if (new_Ny != Ny):
        img = img[:,0:new_Ny,:]

    if verbose:
        print(f'Cropping original (Nx,Ny)=({Nx},{Ny}) to ({new_Nx},{new_Ny})')

    return(img)

## ============================================================
def augment_img_colordiff(img1, img2):
    ## Calculate the difference between two color images, and amplify the color differences to make them easy to see.

    diff = zeros(img1.shape, 'float32')
    diff = float32(img1) - float32(img2)
    diff -= amin(diff)
    diff = 255.0 * diff / amax(diff)
    return(uint8(diff))

## ============================================================
def naive_bayer_recon(raw_img, origin='G', upsample=False):
    ## From a Bayer-sampled raw image, map every other pixel into the output registered color image (i.e. datacube).
    ## This will produce an image that is half the size of the original. However, if choosing "upsample=True", then
    ## the algorithm is followed by a 2x upsampling step to recover the original image size. (Which makes comparison
    ## far easier!)

    (Nx,Ny) = raw_img.shape
    out = zeros((Nx//2,Ny//2,3), raw_img.dtype)
    (Nx_out,Ny_out,_) = out.shape

    if (origin == 'G'):
        red = raw_img[1::2,0::2]
        green1 = raw_img[0::2,0::2]
        green2 = raw_img[1::2,1::2]
        blue = raw_img[0::2,1::2]
    elif (origin == 'R'):
        red = raw_img[0::2,0::2]
        green1 = raw_img[0::2,1::2]
        green2 = raw_img[1::2,0::2]
        blue = raw_img[1::2,1::2]

    if (red.shape != out.shape):
        red = red[:Nx_out,:Ny_out]
    if (green1.shape != out.shape):
        green1 = green1[:Nx_out,:Ny_out]
    if (green2.shape != out.shape):
        green2 = green2[:Nx_out,:Ny_out]
    if (blue.shape != out.shape):
        blue = blue[:Nx_out,:Ny_out]

    out[:,:,0] = red
    out[:,:,1] = ((float32(green1) + float32(green2)) / 2.0).astype(raw_img.dtype)
    out[:,:,2] = blue

    ## Naive-sampling will naturally lead to having half as many pixels as the original image had.
    ## If we then upsample by a factor of two, then we can recover the original image size.
    if upsample:
        out = resize(out, (Nx,Ny,3))

    return(out)

## ============================================================
def generate_bayer_modulation_functions(Nx, Ny, origin='G', show=False):
    ## Generate the three sampling modulation functions for the Bayer color filter array.
    (mm,nn) = indices((Nx,Ny))

    if (origin == 'R'):     ## red is at the origin pixel
        mu_r = 0.25 * (1.0 + cos(pi * mm)) * (1.0 + cos(pi * nn))
        mu_g = 0.5 * (1.0 - cos(pi * mm) * cos(pi * nn))
        mu_b = 0.25 * (1.0 - cos(pi * mm)) * (1.0 - cos(pi * nn))
    elif (origin == 'G'):   ## green is at the origin pixel
        mu_r = 0.25 * (1.0 - cos(pi * mm)) * (1.0 + cos(pi * nn))
        mu_g = 0.5 * (1.0 + cos(pi * mm) * cos(pi * nn))
        mu_b = 0.25 * (1.0 + cos(pi * mm)) * (1.0 - cos(pi * nn))

    if show:
        plt.figure('mu_r[:6,:6]')
        plt.imshow(mu_r[:6,:6])
        plt.colorbar()

        plt.figure('mu_g[:6,:6]')
        plt.imshow(mu_g[:6,:6])
        plt.colorbar()

        plt.figure('mu_b[:6,:6]')
        plt.imshow(mu_b[:6,:6])
        plt.colorbar()

        mu_all = zeros((Nx,Ny,3), 'uint8')
        mu_all[abs(mu_r - 1.0) < 0.1, 0] = 255
        mu_all[abs(mu_g - 1.0) < 0.1, 1] = 255
        mu_all[abs(mu_b - 1.0) < 0.1, 2] = 255

        plt.figure('mu_all[:6,:6]')
        plt.imshow(mu_all[:6,:6])

    return(mu_r, mu_g, mu_b)

## ============================================================
def generate_sampled_image_from_datacube(dcb, sampling_functions, zoom_region=None, show=False):
    ## Given an input registered color image (i.e. a datacube), sample each color plane of the input to simulate
    ## a Bayer-filter-sampled raw image.

    nfunc = len(sampling_functions)
    (Nx,Ny,_) = dcb.shape
    sampled_img = zeros((Nx,Ny), 'float32')
    zoombox = zoom_region

    ## The three "sampling_functions" are for the R, G, and B pixels.
    for i in range(nfunc):
        sampled_img += dcb[:,:,i] * sampling_functions[i]

    if show:
        plt.figure('raw_sampled_img')
        plt.imshow(sampled_img)
        plt.colorbar()

        color_sampled_img = zeros((Nx,Ny,nfunc), 'uint8')
        for i in range(nfunc):
            color_sampled_img[:,:,i] = dcb[:,:,i] * sampling_functions[i]

        plt.figure('color-coded_sampled_img')
        plt.imshow(color_sampled_img)

        if zoom_region is not None:
            plt.figure('color-coded_sampled_img_zoom')
            plt.imshow(color_sampled_img[zoombox[0]:zoombox[1],zoombox[2]:zoombox[3]])
            plt.axis('off')

    return(uint8(sampled_img))

## ============================================================
def fourier_bayer_recon(raw_img, origin='G', masktype='rect', show=False):
    ## Use the Fourier shift-and-mask approach to reconstructing the Bayer-sampled image.

    fft_img = fftshift(fft2(raw_img))
    (Nx,Ny) = fft_img.shape
    print('Nx,Ny=', Nx, Ny)

    mask = create_mask_function(Nx, Ny, Nx//2, Ny//2, masktype=masktype, show=False)
    (Px,Py) = (Nx//2, Ny//2)

    ## Here is the Fourier reconstruction code. We shift the Fourier-domain image, mask it, then inverse transform.
    ## Finally, we combine the channels to reconstruct individual color channels from the luminance and chrominance
    ## images.
    c00 = real(ifft2(ifftshift(fft_img * mask)))
    c01 = real(ifft2(ifftshift(roll(fft_img, -Px, axis=0) * mask)))
    c10 = real(ifft2(ifftshift(roll(fft_img, -Py, axis=1) * mask)))
    c11 = real(ifft2(ifftshift(roll(roll(fft_img, -Py, axis=1), -Px, axis=0) * mask)))

    if (origin == 'R'):     ## red at (0,0)
        G = c00 - c11
        R = c00 + c11 + 2.0 * c01
        B = c00 + c11 - 2.0 * c10
    elif (origin == 'G'):   ## green at (0,0)
        G = c00 + c11
        R = c00 - c11 - 2.0 * c01
        B = c00 - c11 - 2.0 * c10

    ## Finally, insert the three color planes into a single integrated color image (i.e. "datacube").
    out = zeros((Nx,Ny,3), 'float32')
    out[:,:,0] = R
    out[:,:,1] = G
    out[:,:,2] = B

    if show:
        ## If we use "f2abs = log(abs(fft_img))" then we may get divide-by-zero warnings. To prevent this, use "where".
        f2abs = where(abs(fft_img)>0, log(abs(fft_img)), 0)
        plt.figure('log(abs(fft_img))')
        plt.imshow(f2abs, extent=[-1,1,-1,1], vmin=0, vmax=17, aspect='auto')
        plt.xticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        plt.yticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        plt.xlabel('x-axis frequencies (Nyquist units)')
        plt.ylabel('y-axis frequencies (Nyquist units)')
        plt.colorbar()

        ## The lines showing where the cross-sections are taken from.
        plt.plot([-1.0,1.0], [0.0,0.0], 'r--')
        plt.plot([0.0,0.0], [-1.0,1.0], 'b--')
        plt.plot([-1.0,1.0], [-1.0,1.0], 'g--')

        ## Next draw circles around the center of each channel.
        draw_bayer_fft_circles(Nx,Ny, normalize=True)

        ## Get the pixels along the image diagonal, so we can take a diagonal cross-section.
        (mm,nn) = indices((Nx,Ny))
        r = sqrt((mm-(Nx/2.0))**2 + (nn-(Ny/2.0))**2)
        diagonal_line_mask = (mm == uint(nn * Nx / Ny))
        Ndiag = sum(diagonal_line_mask)
        diag_dist = r[diagonal_line_mask]
        diag_dist[arange(Ndiag) < (Ndiag/2)] *= -1.0

        plt.figure('log(abs(fft_img))_cross-section')
        plt.plot(arange(Ny)-(Ny/2), f2abs[Nx//2,:], alpha=0.75, label='horiz slice')
        plt.plot(arange(Nx)-(Nx/2), f2abs[:,Ny//2], alpha=0.75, label='vert slice')
        plt.plot(diag_dist, f2abs[diagonal_line_mask], alpha=0.75, label='diag slice')
        plt.xlabel('pixel distance along frequency axis')
        plt.ylabel('Fourier domain absolute value (log scale)')
        plt.legend()

        plt.figure('fourier_domain_mask')
        plt.imshow(mask)
        plt.colorbar()

    return(out)

## ===============================================================================================
def read_sony_image(filename):
    with open(filename, 'rb') as fileobj:
        raw = fileobj.read()

    Ny = struct.unpack(('<1H').encode('ascii'), raw[0:2])[0]
    Nx = struct.unpack(('<1H').encode('ascii'), raw[2:4])[0]
    #print('(Nx,Ny)=',Nx,Ny)

    buffersize = len(raw) - 4
    bytesize = 2
    img = zeros((Nx,Ny), 'uint16')

    q = 4
    xvalues = Nx - 1 - arange(Nx)
    for x in xvalues:
        try:
            row = uint16(struct.unpack(('<'+str(Ny)+'H').encode('ascii'), raw[q:q+(Ny*bytesize)]))
        except Exception as e:
            raise ImportError('Cannot decode datacube: ' + repr(e))
        img[x,:] = row
        q += Ny * bytesize

    img = img[8:,:]
    img = img[:,1:]
    nsat = sum(img >= 4095)
    if nsat > 0:
        print('nsat=', nsat)

    return(img)

## ============================================================
def truncate_rgb_float_to_uint8(floatimg):
    rgb_img = array(floatimg)
    rgb_img[rgb_img > 255.0] = 255.0
    rgb_img[rgb_img < 0.0] = 0.0
    rgb_img = uint8(rgb_img)
    return(rgb_img)

## ============================================================
def draw_quadbayer_fft_circles(Nx, Ny, normalize=False):
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)

    if normalize:
        ## Plot the circles in a frequency domain of Nyquist units
        radius = 0.25
        cen1 = ( 0, 0)
        cen2 = ( 0.5, 0)
        cen3 = (-0.5, 0)
        cen4 = ( 0,-0.5)
        cen5 = ( 0, 0.5)
        cen6 = ( 0.5, 0.5)
        cen7 = (-0.5,-0.5)
        cen8 = (-0.5, 0.5)
        cen9 = ( 0.5,-0.5)
    else:
        ## Plot the circles in a frequency domain of inverse pixel units
        radius = (Px-Mx) / 2
        cen1 = ((Nx-1)/4, (Ny-1)/4)
        cen2 = ((Nx-1)/2, (Ny-1)/4)
        cen3 = (0.0, (Ny-1)/4)
        cen4 = ((Nx-1)/4, 0.0)
        cen5 = ((Nx-1)/4, (Ny-1)/2)
        cen6 = ((Nx-1)/2, (Ny-1)/2)
        cen7 = (0.0, 0.0)
        cen8 = (0.0, (Ny-1)/2)
        cen9 = ((Nx-1)/2, 0.0)

    centers = [cen1, cen2, cen3, cen4, cen5, cen6, cen7, cen8, cen9]
    thetas = linspace(0.0, 2.0*pi, 200)
    colors = ['k','b','b','r','r','g','g','g','g']

    for i in range(len(centers)):
        xcircle = centers[i][1] + radius * cos(thetas)
        ycircle = centers[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, '-', color=colors[i], lw=3)

    return

## ============================================================
def generate_quadbayer_modulation_functions(Nx, Ny, origin='G', show=False):
    (mm,nn) = indices((Nx,Ny))
    mu_x = sqrt(2.0) * cos(0.25 * pi * (2.0*mm - 1.0))
    mu_y = sqrt(2.0) * cos(0.25 * pi * (2.0*nn - 1.0))

    if (origin == 'R'):
        mu_g = 0.5 - 0.5 * mu_x * mu_y
        mu_r = 0.25 * (1.0 + mu_x) * (1.0 + mu_y)
        mu_b = 0.25 * (1.0 - mu_x) * (1.0 - mu_y)
    elif (origin == 'G'):
        mu_g = 0.5 + 0.5 * mu_x * mu_y
        mu_r = 0.25 * (1.0 - mu_x) * (1.0 + mu_y)
        mu_b = 0.25 * (1.0 + mu_x) * (1.0 - mu_y)

    if show:
        plt.figure('mu_r')
        plt.imshow(mu_r[:16,:16])
        plt.colorbar()

        plt.figure('mu_g')
        plt.imshow(mu_g[:16,:16])
        plt.colorbar()

        plt.figure('mu_b')
        plt.imshow(mu_b[:16,:16])
        plt.colorbar()

        mu_all = zeros((Nx,Ny,3), 'uint8')
        mu_all[abs(mu_r - 1.0) < 0.1, 0] = 255
        mu_all[abs(mu_g - 1.0) < 0.1, 1] = 255
        mu_all[abs(mu_b - 1.0) < 0.1, 2] = 255

        plt.figure('all_color_mod')
        plt.imshow(mu_all[:16,:16])

    return(mu_r, mu_g, mu_b)

## ============================================================
def naive_quadbayer_recon(raw_img, origin='G', upsample=False):
    ## From a Bayer-sampled raw image, map every other 2x2 superpixel into the output registered color image
    ## (i.e. datacube). This will produce an image that is half the size of the original. However, if choosing
    ## "upsample=True", then the algorithm is followed by a 2x upsampling step to recover the original image size.
    ## (Which makes comparison with the original far easier!)

    (Nx,Ny) = raw_img.shape
    out = zeros((Nx//2,Ny//2,3), raw_img.dtype)
    (Nx_out,Ny_out,_) = out.shape

    red = zeros((Nx_out,Ny_out), raw_img.dtype)
    green1 = zeros((Nx_out,Ny_out), raw_img.dtype)
    green2 = zeros((Nx_out,Ny_out), raw_img.dtype)
    blue = zeros((Nx_out,Ny_out), raw_img.dtype)

    if (origin == 'G'):
        red[0::2,0::2] = raw_img[2::4,0::4]
        red[1::2,0::2] = raw_img[3::4,0::4]
        red[0::2,1::2] = raw_img[2::4,1::4]
        red[1::2,1::2] = raw_img[3::4,1::4]

        green1[0::2,0::2] = raw_img[0::4,0::4]
        green1[1::2,0::2] = raw_img[1::4,0::4]
        green1[0::2,1::2] = raw_img[0::4,1::4]
        green1[1::2,1::2] = raw_img[1::4,1::4]

        green2[0::2,0::2] = raw_img[2::4,2::4]
        green2[1::2,0::2] = raw_img[3::4,2::4]
        green2[0::2,1::2] = raw_img[2::4,3::4]
        green2[1::2,1::2] = raw_img[3::4,3::4]

        blue[0::2,0::2] = raw_img[0::4,2::4]
        blue[1::2,0::2] = raw_img[1::4,2::4]
        blue[0::2,1::2] = raw_img[0::4,3::4]
        blue[1::2,1::2] = raw_img[1::4,3::4]
    elif (origin == 'R'):
        red[0::2,0::2] = raw_img[0::4,0::4]
        red[1::2,0::2] = raw_img[1::4,0::4]
        red[0::2,1::2] = raw_img[0::4,1::4]
        red[1::2,1::2] = raw_img[1::4,1::4]

        green1[0::2,0::2] = raw_img[2::4,0::4]
        green1[1::2,0::2] = raw_img[3::4,0::4]
        green1[0::2,1::2] = raw_img[2::4,1::4]
        green1[1::2,1::2] = raw_img[3::4,1::4]

        green2[0::2,0::2] = raw_img[0::4,2::4]
        green2[1::2,0::2] = raw_img[1::4,2::4]
        green2[0::2,1::2] = raw_img[0::4,3::4]
        green2[1::2,1::2] = raw_img[1::4,3::4]

        blue[0::2,0::2] = raw_img[2::4,2::4]
        blue[1::2,0::2] = raw_img[3::4,2::4]
        blue[0::2,1::2] = raw_img[2::4,3::4]
        blue[1::2,1::2] = raw_img[3::4,3::4]

    if (red.shape != out.shape):
        red = red[:Nx_out,:Ny_out]
    if (green1.shape != out.shape):
        green1 = green1[:Nx_out,:Ny_out]
    if (green2.shape != out.shape):
        green2 = green2[:Nx_out,:Ny_out]
    if (blue.shape != out.shape):
        blue = blue[:Nx_out,:Ny_out]

    out[:,:,0] = red
    out[:,:,1] = ((float32(green1) + float32(green2)) / 2.0).astype(raw_img.dtype)
    out[:,:,2] = blue

    ## Naive-sampling will naturally lead to having half as many pixels as the original image had.
    ## If we then upsample by a factor of two, then we can recover the original image size.
    if upsample:
        out = resize(out, (Nx,Ny,3))    ## this can break with odd-number dimension sizes ... what function allows noninteger dimension scaling?

    return(out)

## ============================================================
def fourier_quadbayer_recon(raw_img, origin='G', masktype='rect', show=False):
    ## Use the Fourier shift-and-mask approach to reconstructing the Bayer-sampled image.
    fft_img = fftshift(fft2(raw_img))

    (Nx,Ny) = raw_img.shape
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)
    mask = create_mask_function(Nx, Ny, Nx//4, Ny//4, masktype=masktype)

    c00 = real(ifft2(ifftshift(fft_img * mask)))
    c10 = real(ifft2(ifftshift(roll(fft_img/(1+1j), Mx, axis=0) * mask)))
    c01 = real(ifft2(ifftshift(roll(fft_img/(1+1j), My, axis=1) * mask)))
    c11 = real(ifft2(ifftshift(roll(roll(fft_img/(2.0j), My, axis=1), Mx, axis=0) * mask)))

    if (origin == 'R'):     ## red at (0,0)
        G = c00 - 2*c11
        R = c00 + 2*c11 + 2*c10 + 2*c01
        B = c00 + 2*c11 - 2*c10 - 2*c01
    elif (origin == 'G'):   ## green at (0,0)
        G = c00 + 2*c11
        R = c00 - 2*c11 - 2*c10 + 2*c01
        B = c00 - 2*c11 + 2*c10 - 2*c01

    out = zeros((Nx,Ny,3), 'float32')
    out[:,:,0] = R
    out[:,:,1] = G
    out[:,:,2] = B

    if show:
        ## If we use "f2abs = log(abs(fft_img))" then we may get divide-by-zero warnings. To prevent this, use "where".
        f2abs = where(abs(fft_img)>0, log(abs(fft_img)), 0)

        plt.figure('log(abs(fft_img))')
        plt.imshow(f2abs, extent=[-1,1,-1,1], aspect='auto')
        plt.xlabel('x-axis frequencies (Nyquist units)')
        plt.ylabel('y-axis frequencies (Nyquist units)')
        plt.xticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        plt.yticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        plt.colorbar()

        ## The lines showing where the cross-sections are taken from.
        plt.plot([-1.0,1.0], [0.0,0.0], 'r--')
        plt.plot([0.0,0.0], [-1.0,1.0], 'b--')
        plt.plot([-1.0,1.0], [-1.0,1.0], 'g--')

        draw_quadbayer_fft_circles(Nx, Ny, normalize=True)

        ## Get the pixels along the image diagonal, so we can take a diagonal cross-section.
        (mm,nn) = indices((Nx,Ny))
        r = sqrt((mm-(Nx/2))**2 + (nn-(Ny/2))**2)
        diagonal_line_mask = (mm == uint(nn * Nx / Ny))
        Ndiag = sum(diagonal_line_mask)
        diag_dist = r[diagonal_line_mask]
        diag_dist[arange(Ndiag) < (Ndiag/2)] *= -1.0

        #plt.figure('log(abs(c00))')
        #plt.imshow(log(abs(fft_img)))
        #plt.colorbar()

        #plt.figure('log(abs(c01))')
        #plt.imshow(log(abs(roll(fft_img, -Mx, axis=0))))
        #plt.colorbar()

        #plt.figure('log(abs(c10))')
        #plt.imshow(log(abs(roll(fft_img, -My, axis=1))))
        #plt.colorbar()

        plt.figure('log(abs(fft_img))_cross-section')
        plt.plot(arange(Ny)-(Ny/2), f2abs[Nx//2,:], 'r-', alpha=0.75, label='horiz slice')
        plt.plot(arange(Nx)-(Nx/2), f2abs[:,Ny//2], 'b-', alpha=0.75, label='vert slice')
        plt.plot(diag_dist, f2abs[diagonal_line_mask], 'g-', alpha=0.75, label='diag slice')
        plt.xlabel('pixel distance along frequency axis')
        plt.ylabel('Fourier domain absolute value (log scale)')
        plt.legend()

        plt.figure('mask')
        plt.imshow(mask)
        plt.colorbar()

    return(out)

## ============================================================
def simulate_quadbayer_rawimg_from_dcb(filename, origin='G', binning=1, blurring=1, show=False):
    dcb = imread('./images/'+filename)[::-1,:,:]    ## flip images to match to image origin='lower'
    if (binning > 1):
        dcb = image_binning(dcb)

    ## Make sure that the image size is an even factor of two, for ease of sampling.
    dcb = evencrop(dcb)

    if blurring > 1:
        dcb = image_blur(dcb, 'gaussian', blurring, show_image=show)

    (Nx,Ny,_) = dcb.shape
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)  ## mask size
    #print(f'(Nx,Ny)=({Nx},{Ny}), (Px,Py)=({Px},{Py}), (Mx,My)=({Mx},{My})')

    if show:
        plt.figure('original_dcb')
        plt.imshow(dcb)

    (mu_r, mu_g, mu_b) = generate_quadbayer_modulation_functions(Nx, Ny, origin=origin, show=show)
    raw_img = (dcb[:,:,0] * mu_r) + (dcb[:,:,1] * mu_g) + (dcb[:,:,2] * mu_b)

    if show:
        raw_img_rgb = zeros_like(dcb)
        raw_img_rgb[:,:,0] = dcb[:,:,0] * mu_r
        raw_img_rgb[:,:,1] = dcb[:,:,1] * mu_g
        raw_img_rgb[:,:,2] = dcb[:,:,2] * mu_b

        plt.figure('raw_sampled_img_colorized')
        plt.imshow(raw_img_rgb)

    return(dcb, raw_img)

## ===============================================================================================
def read_binary_image(filename, Nx=2048, Ny=2448):
    with open(filename, 'rb') as fileobj:
        raw = fileobj.read()

    buffersize = len(raw)
    bytesize = 1
    img = zeros((Nx,Ny), 'uint16')
    img_nbytes = Nx * Ny * bytesize

    q = 0
    xvalues = Nx - 1 - arange(Nx)
    for x in xvalues:
        try:
            row = uint16(struct.unpack(('<'+str(Ny)+'B').encode('ascii'), raw[q:q+(Ny*bytesize)]))
        except Exception as e:
            raise ImportError('Cannot decode datacube: ' + repr(e))
        img[x,:] = row
        q += Ny * bytesize

    nsat = sum(img == 255)
    if nsat > 0:
        print('nsat=', nsat, ' pixels')

    return(img)

## ===========================================================================================
def naive_monopol_recon(img, config='0-45-90-135', upsample=False):
    (Nx,Ny) = img.shape
    s0 = zeros((Nx//2,Ny//2), 'float32')
    (Nx_out,Ny_out) = s0.shape

    if (config == '0-45-90-135'):
        img0 = float32(img[0::2,0::2])
        img45 = float32(img[0::2,1::2])
        img90 = float32(img[1::2,1::2])
        img135 = float32(img[1::2,0::2])
    elif (config == '90-45-0-135'):
        img0 = float32(img[1::2,1::2])
        img45 = float32(img[0::2,1::2])
        img90 = float32(img[0::2,0::2])
        img135 = float32(img[1::2,0::2])
    elif (config == '135-0-45-90'):
        img0 = float32(img[0::2,1::2])
        img45 = float32(img[1::2,1::2])
        img90 = float32(img[1::2,0::2])
        img135 = float32(img[0::2,0::2])

    if (img0.shape != s0.shape):
        img0 = img0[:Nx_out,:Ny_out]
    if (img45.shape != s0.shape):
        img45 = img45[:Nx_out,:Ny_out]
    if (img90.shape != s0.shape):
        img90 = img90[:Nx_out,:Ny_out]
    if (img135.shape != s0.shape):
        img135 = img135[:Nx_out,:Ny_out]

    s0 = (img0 + img45 + img90 + img135) / 4.0
    s1 = 0.5 * (img0 - img90)
    s2 = 0.5 * (img45 - img135)

    (ns1, ns2) = calc_normstokes(s0, s1, s2)

    ## Naive-sampling will naturally lead to having half as many pixels as the original image had.
    ## If we then upsample by a factor of two, then we can recover the original image size.
    if upsample:
        s0 = resize(s0, (Nx,Ny))
        ns1 = resize(ns1, (Nx,Ny))
        ns2 = resize(ns2, (Nx,Ny))

    return(s0, ns1, ns2)

## ===============================================================================================
def fourier_monopol_recon(img, config='0-45-90-135', masktype='rect', show=False):
    (Nx,Ny) = img.shape
    (Px,Py) = (Nx//2, Ny//2)
    mask = create_mask_function(Nx, Ny, Nx//2, Ny//2, masktype=masktype, show=False)

    fft_img = fftshift(fft2(img))          ## s0 component
    fft_horiz = roll(fft_img, Py, axis=1)   ## s1+s2 component
    fft_vert = roll(fft_img, Px, axis=0)    ## s1-s2 component

    C00 = fft_img * mask
    C10 = fft_horiz * mask
    C01 = fft_vert * mask

    c_01 = real(ifft2(ifftshift(C01)))
    c_10 = real(ifft2(ifftshift(C10)))

    s0 = real(ifft2(ifftshift(C00)))

    if (config == '0-45-90-135'):
        s1 = c_01 - c_10
        s2 = c_01 + c_10
    elif (config == '90-45-0-135'):
        s1 = NaN
        s2 = NaN
        raise NotImplementedError
    elif (config == '135-0-45-90'):
        s1 = c_01 - c_10
        s2 = -c_01 - c_10

    (ns1, ns2) = calc_normstokes(s0, s1, s2)

    ## Finally, show the Fourier-domain magnitude image, with circular regions drawn.
    plt.figure('fft_img')
    plt.imshow(log(abs(fft_img)), extent=[-1,1,-1,1], vmin=0, vmax=17, aspect='auto')
    plt.xticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
    plt.yticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
    plt.xlabel('x-axis frequencies (Nyquist units)')
    plt.ylabel('y-axis frequencies (Nyquist units)')
    plt.colorbar()
    draw_monopol_fft_circles(Nx, Ny, alpha=0.5)

    return(s0, ns1, ns2)

## ===============================================================================================
def generate_rgbpol_modulation_functions(Nx, Ny, origin='G', config='0-45-90-135', show=False):
    (mm,nn) = indices((Nx,Ny))
    mu_x = sqrt(2.0) * cos(0.25 * pi * (2.0*mm - 1.0))
    mu_y = sqrt(2.0) * cos(0.25 * pi * (2.0*nn - 1.0))

    if (origin == 'R'):
        mu_r = 0.25 * (1.0 + mu_x) * (1.0 + mu_y)
        mu_g = 0.5 * (1.0 - mu_x * mu_y)
        mu_b = 0.25 * (1.0 - mu_x) * (1.0 - mu_y)
    elif (origin == 'G'):
        mu_r = 0.25 * (1.0 - mu_x) * (1.0 + mu_y)
        mu_g = 0.5 * (1.0 + mu_x * mu_y)
        mu_b = 0.25 * (1.0 + mu_x) * (1.0 - mu_y)

    print('detected config:', config)

    if (config == None) or (config == '0-45-90-135'):
        mu_splus = cos(pi * mm)
        mu_sminus = cos(pi * nn)
    elif (config == '90-45-0-135'):
        raise NotImplementedError
    elif (config == '135-0-45-90'):
        mu_splus = -cos(pi * nn)
        mu_sminus = cos(pi * mm)

    mu_pol0 = abs(0.5 * (mu_splus + mu_sminus) - 1.0) < 1.0e-7
    mu_pol90 = abs(0.5 * (mu_splus + mu_sminus) + 1.0) < 1.0e-7
    mu_pol45 = abs(0.5 * (mu_splus - mu_sminus) - 1.0) < 1.0e-7
    mu_pol135 = abs(0.5 * (mu_splus - mu_sminus) + 1.0) < 1.0e-7

    mu_r_0 = abs(mu_r * mu_pol0) > 1.0e-7
    mu_r_45 = abs(mu_r * mu_pol45) > 1.0e-7
    mu_r_90 = abs(mu_r * mu_pol90) > 1.0e-7
    mu_r_135 = abs(mu_r * mu_pol135) > 1.0e-7

    mu_g_0 = abs(mu_g * mu_pol0) > 1.0e-7
    mu_g_45 = abs(mu_g * mu_pol45) > 1.0e-7
    mu_g_90 = abs(mu_g * mu_pol90) > 1.0e-7
    mu_g_135 = abs(mu_g * mu_pol135) > 1.0e-7

    mu_b_0 = abs(mu_b * mu_pol0) > 1.0e-7
    mu_b_45 = abs(mu_b * mu_pol45) > 1.0e-7
    mu_b_90 = abs(mu_b * mu_pol90) > 1.0e-7
    mu_b_135 = abs(mu_b * mu_pol135) > 1.0e-7

    all_mu = [mu_r_0, mu_r_45, mu_r_90, mu_r_135, mu_g_0, mu_g_45, mu_g_90, mu_g_135, mu_b_0, mu_b_45, mu_b_90, mu_b_135]
    other_mu = [mu_r, mu_g, mu_b, mu_splus, mu_sminus, mu_pol0, mu_pol90, mu_pol45, mu_pol135]

    if show:
        labels = ['mu_r_0', 'mu_r_45', 'mu_r_90', 'mu_r_135', 'mu_g_0', 'mu_g_45', 'mu_g_90', 'mu_g_135',
                  'mu_b_0', 'mu_b_45', 'mu_b_90', 'mu_b_135']

        #for i in range(12):
        #    plt.figure(labels[i])
        #    plt.imshow(all_mu[i][:16,:16])
        #    plt.colorbar()

        mu_rgb = zeros((Nx,Ny,3), 'uint8')
        mu_rgb[mu_r_0,0] = 0
        mu_rgb[mu_r_45,0] = 45
        mu_rgb[mu_r_90,0] = 90
        mu_rgb[mu_r_135,0] = 135
        mu_rgb[mu_g_0,1] = 0
        mu_rgb[mu_g_45,1] = 45
        mu_rgb[mu_g_90,1] = 90
        mu_rgb[mu_g_135,1] = 135
        mu_rgb[mu_b_0,2] = 0
        mu_rgb[mu_b_45,2] = 45
        mu_rgb[mu_b_90,2] = 90
        mu_rgb[mu_b_135,2] = 135

        plt.figure('all_color_mod')
        plt.imshow(mu_rgb[:16,:16])

    return(all_mu, other_mu)

## ===============================================================================================
def simulate_rgbpol_rawimg_from_dcb(filename, pol='None', origin='G', binning=1, blurring=1, polconfig=None, show=False):
    dcb = imread('./images/'+filename)[::-1,:,:]    ## flip images to match to image origin='lower'
    if (binning > 1):
        dcb = image_binning(dcb)

    ## Make sure that the image size is an even factor of two, for ease of sampling.
    dcb = evencrop(dcb)

    if blurring > 1:
        dcb = image_blur(dcb, 'gaussian', blurring, show_image=show)

    (Nx,Ny,_) = dcb.shape
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)  ## mask size

    if show:
        plt.figure('original_dcb')
        plt.imshow(dcb)

    (all_mu, other_mu) = generate_rgbpol_modulation_functions(Nx, Ny, origin=origin, config=polconfig, show=show)

    if (pol == 'None'):
        pol0_filter = 0.5
        pol45_filter = 0.5
        pol90_filter = 0.5
        pol135_filter = 0.5
    elif is_number(pol):
        pol_angle_radians = float32(pol) * pi / 180.0
        pol0_filter = cos(pol_angle_radians)**2
        pol45_filter = cos(pol_angle_radians - pi/4.0)**2
        pol90_filter = cos(pol_angle_radians - pi/2.0)**2
        pol135_filter = cos(pol_angle_radians - 3.0*pi/4.0)**2
    else:
        raise ValueError(f'pol = "{pol}" is not valid. Only "None" or a number is allowed.')

    R = dcb[:,:,0]
    G = dcb[:,:,1]
    B = dcb[:,:,2]

    raw_img = (R * pol0_filter * all_mu[0])
    raw_img += (R * pol45_filter * all_mu[1])
    raw_img += (R * pol90_filter * all_mu[2])
    raw_img += (R * pol135_filter * all_mu[3])

    raw_img += (G * pol0_filter * all_mu[4])
    raw_img += (G * pol45_filter * all_mu[5])
    raw_img += (G * pol90_filter * all_mu[6])
    raw_img += (G * pol135_filter * all_mu[7])

    raw_img += (B * pol0_filter * all_mu[8])
    raw_img += (B * pol45_filter * all_mu[9])
    raw_img += (B * pol90_filter * all_mu[10])
    raw_img += (B * pol135_filter * all_mu[11])

    raw_img = uint8(raw_img)

    if show:
        plt.figure('simulated raw image rgbpol')
        plt.imshow(raw_img)
        plt.colorbar()

        raw_img_colorized = zeros_like(dcb)
        raw_img_colorized[:,:,0] = raw_img * other_mu[0]
        raw_img_colorized[:,:,1] = raw_img * other_mu[1]
        raw_img_colorized[:,:,2] = raw_img * other_mu[2]

        plt.figure('simulated raw image - colorized')
        plt.imshow(raw_img_colorized)

    return(dcb, raw_img)

## ===============================================================================================
def create_mask_function(Nx, Ny, Mx, My, masktype, show=False):
    (Px,Py) = (Nx//2, Ny//2)
    mask = zeros((Nx,Ny), 'float32')

    if (masktype == 'rect'):
        mask[Px-(Mx//2):Px+(Mx//2), Py-(My//2):Py+(My//2)] = 1.0
        return(mask)
    elif (masktype == 'hanning'):
        wx = hanning(Mx)
        wy = hanning(My)
    elif (masktype == 'hamming'):
        wx = hamming(Mx)
        wy = hamming(My)
    elif (masktype == 'blackman'):
        wx = blackman(Mx)
        wy = blackman(My)
    elif (masktype == 'supergauss'):
        from scipy import signal
        wx = signal.general_gaussian(Mx, p=3.0, sig=Mx/2.9)
        wy = signal.general_gaussian(My, p=3.0, sig=My/2.9)

    window2d = outer(wx,wy)
    mask[Px-(Mx//2):Px+(Mx//2), Py-(My//2):Py+(My//2)] = window2d

    if show:
        plt.figure('window_function_1d')
        plt.plot(linspace(-(Mx//2),+(Mx//2)-1,Mx), wx)
        plt.plot(linspace(-(My//2),+(My//2)-1,My), wy)

        (xx,yy) = indices((Nx,Ny))
        plt.figure('window_function_2d')
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(xx, yy, mask, lw=0.75)

    return(mask)

## ===========================================================================================
def naive_rgbpol_recon(img, origin='G', config='0-45-90-135', upsample=False):
    (Nx,Ny) = img.shape
    (all_s0, all_ns1, all_ns2) = naive_monopol_recon(img, config=config, upsample=False)
    all_s0 *= 2
    rgb_s0 = naive_bayer_recon(all_s0, origin=origin, upsample=False)
    rgb_ns1 = naive_bayer_recon(all_ns1, origin=origin, upsample=False)
    rgb_ns2 = naive_bayer_recon(all_ns2, origin=origin, upsample=False)

    ## Naive-sampling will naturally lead to having half as many pixels as the original image had.
    ## If we then upsample by a factor of two, then we can recover the original image size.
    if upsample:
        rgb_s0 = resize(rgb_s0, (Nx,Ny,3))
        rgb_ns1 = resize(rgb_ns1, (Nx,Ny,3))
        rgb_ns2 = resize(rgb_ns2, (Nx,Ny,3))

    return(rgb_s0, rgb_ns1, rgb_ns2)

## ===========================================================================================
def calc_normstokes(s0, s1, s2):
    if any(s0 < 1.0e-7):
        okay = (s0 > 1.0e-7)
        ns1 = zeros_like(s0)
        ns1[okay] = s1[okay] / s0[okay]
        ns2 = zeros_like(s0)
        ns2[okay] = s2[okay] / s0[okay]
    else:
        ns1 = s1 / s0
        ns2 = s2 / s0

    return(ns1, ns2)

## ===========================================================================================
def fourier_rgbpol_recon(img, origin='G', config='0-45-90-135', masktype='rect', show=False):
    (Nx,Ny) = img.shape
    (Px,Py) = (Nx//2, Ny//2)
    (Mx,My) = (Px//2, Py//2)  ## mask size
    #print(f'(Nx,Ny) = ({Nx},{Ny}),  (Px,Py) = ({Px},{Py}),  (Mx,My) = ({Mx},{My})')

    f2 = fftshift(fft2(img))

    if show:
        f2abs = where(abs(f2)>0, log(abs(f2)), 0)
        plt.figure('log(abs(fft_img))')
        plt.imshow(f2abs, extent=[-1,1,-1,1], aspect='auto')
        plt.xticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        plt.yticks([-1,-0.5,0,0.5,1], ['-1','-1/2','0','1/2','1'])
        draw_rgbpol_fft_circles()
        plt.colorbar()
        plt.xlabel('x-axis frequencies (Nyquist units)')
        plt.ylabel('y-axis frequencies (Nyquist units)')

    mask = create_mask_function(Nx, Ny, Nx//4, Ny//4, masktype)

    c00 = ifft2(ifftshift(f2 * mask))
    c10 = ifft2(ifftshift(roll(f2, -Mx, axis=0) * mask))
    cm10 = ifft2(ifftshift(roll(f2, Mx, axis=0) * mask))
    c01 = ifft2(ifftshift(roll(f2, -My, axis=1) * mask))
    c0m1 = ifft2(ifftshift(roll(f2, My, axis=1) * mask))
    c02 = ifft2(ifftshift(roll(f2, -Py, axis=1) * mask))
    c20 = ifft2(ifftshift(roll(f2, Px, axis=0) * mask))
    c11 = ifft2(ifftshift(roll(roll(f2, -Mx, axis=0), -My, axis=1) * mask))
    cm1m1 = ifft2(ifftshift(roll(roll(f2, Mx, axis=0), My, axis=1) * mask))
    c1m1 = ifft2(ifftshift(roll(roll(f2, -Mx, axis=0), My, axis=1) * mask))
    cm11 = ifft2(ifftshift(roll(roll(f2, Mx, axis=0), -My, axis=1) * mask))
    c12 = ifft2(ifftshift(roll(roll(f2, -Mx, axis=0), -Py, axis=1) * mask))
    cm12 = ifft2(ifftshift(roll(roll(f2, Mx, axis=0), -Py, axis=1) * mask))
    c21 = ifft2(ifftshift(roll(roll(f2, -Px, axis=0), -My, axis=1) * mask))
    c2m1 = ifft2(ifftshift(roll(roll(f2, -Px, axis=0), My, axis=1) * mask))

    ## If you want to see the intermediate reconstruction chrominances...
    if False:
        Us0 = 4 * real(c00)
        Usp = 4 * real(c20)
        Usm = 4 * real(c02)
        Vs0 = 2 * real(-1j*c11 - c1m1 - cm11 + 1j*cm1m1)
        Vsp = 2 * real(-c11 + 1j*c1m1 - 1j*cm11 - cm1m1)
        Vsm = 2 * real(-c11 - 1j*c1m1 + 1j*cm11 - cm1m1)
        Ws0 = real((1+1j)*c01 + (1-1j)*c0m1 - (1+1j)*c10 - (1-1j)*cm10)
        Wsp = real((-1+1j)*c10 + (1+1j)*c21 + (1-1j)*c2m1 - (1+1j)*cm10)
        Wsm = real((1-1j)*c01 + (1+1j)*c0m1 - (1+1j)*c12 - (1-1j)*cm12)

        Rs0 = (Us0 + Vs0 + 2*Ws0) / 2
        Gs0 = (Us0 - Vs0) / 2
        Bs0 = (Us0 + Vs0 - 2*Ws0) / 2

        Rsp = (Usp + Vsp + 2*Wsp) / 2
        Gsp = (Usp - Vsp) / 2
        Bsp = (Usp + Vsp - 2*Wsp) / 2

        Rsm = (Usm + Vsm + 2*Wsm) / 2
        Gsm = (Usm - Vsm) / 2
        Bsm = (Usm + Vsm - 2*Wsm) / 2

        if (config == '0-45-90-135'):
            Rs1 = 0.5 * (Rsp + Rsm)
            Rs2 = 0.5 * (Rsp - Rsm)
            Gs1 = 0.5 * (Gsp + Gsm)
            Gs2 = 0.5 * (Gsp - Gsm)
            Bs1 = 0.5 * (Bsp + Bsm)
            Bs2 = 0.5 * (Bsp - Bsm)
        elif (config == '135-0-45-90'):
            Rs1 = 0.5 * (Rsp - Rsm)
            Rs2 = -0.5 * (Rsp + Rsm)
            Gs1 = 0.5 * (Gsp - Gsm)
            Gs2 = -0.5 * (Gsp + Gsm)
            Bs1 = 0.5 * (Bsp - Bsm)
            Bs2 = -0.5 * (Bsp + Bsm)

    if (config == '0-45-90-135'):
        raise NotImplementedError
    elif (config == '135-0-45-90'):
        Rs0 = real(2*c00 - (1+1j)*c10 - c1m1 - (1-1j)*cm10 - cm11 + 1j*cm1m1 - 1j*c11 + (1-1j)*c0m1 + (1+1j)*c01)
        #Rsp = real(2*c20 - c11 + 1j*c1m1 - 1j*cm11 - cm1m1 - (1-1j)*c10 + (1+1j)*c21 + (1-1j)*c2m1 - (1+1j)*cm10)
        #Rsm = real(2*c02 - c11 - 1j*c1m1 + 1j*cm11 - cm1m1 + (1-1j)*c01 + (1+1j)*c0m1 - (1+1j)*c12 - (1-1j)*cm12)
        Rs1 = 0.5 * real(2*c20 - 2*c02 + 2j*c1m1 - 2j*cm11 - (1-1j)*c10 + (1+1j)*c21 + (1-1j)*c2m1 - (1+1j)*cm10
                         - (1-1j)*c01 - (1+1j)*c0m1 + (1+1j)*c12 + (1-1j)*cm12)
        Rs2 = 0.5 * real(-2*c20 - 2*c02 + 2*c11 + 2*cm1m1 + (1-1j)*c10 - (1+1j)*c21 - (1-1j)*c2m1 + (1+1j)*cm10
                         - (1-1j)*c01 - (1+1j)*c0m1 + (1+1j)*c12 + (1-1j)*cm12)

        Gs0 = real((2 * c00) + 1j*c11 + c1m1 + cm11 - 1j*cm1m1)
        #Gsp = real((2 * c20) + c11 - (1j * c1m1) + (1j * cm11) + cm1m1)
        #Gsm = real((2 * c02) + c11 + (1j * c1m1) - (1j * cm11) + cm1m1)
        Gs1 = real(c20 - c02 - 1j*c1m1 + 1j*cm11)
        Gs2 = real(-c20 - c02 - c11 - cm1m1)

        Bs0 = real(2*c00 - 1j*c11 - c1m1 - cm11 + 1j*cm1m1 - (1+1j)*c01 - (1-1j)*c0m1 + (1+1j)*c10 + (1-1j)*cm10)
        #Bsp = real(2*c20 - c11 + 1j*c1m1 - 1j*cm11 - cm1m1 + (1-1j)*c10 - (1+1j)*c21 - (1-1j)*c2m1 + (1+1j)*cm10)
        #Bsm = real(2*c02 - c11 - 1j*c1m1 + 1j*cm11 - cm1m1 - (1-1j)*c01 - (1+1j)*c0m1 + (1+1j)*c12 + (1-1j)*cm12)
        Bs1 = 0.5 * real(2*c20 - 2*c02 + 2j*c1m1 - 2j*cm11 + (1-1j)*c10 - (1+1j)*c21 - (1-1j)*c2m1 + (1+1j)*cm10
                         + (1-1j)*c01 + (1+1j)*c0m1 - (1+1j)*c12 - (1-1j)*cm12)
        Bs2 = 0.5 * real(-2*c20 - 2*c02 + 2*c11 + 2*cm1m1 - (1-1j)*c10 + (1+1j)*c21 + (1-1j)*c2m1 - (1+1j)*cm10
                         + (1-1j)*c01 + (1+1j)*c0m1 - (1+1j)*c12 - (1-1j)*cm12)

    ## Prevent any divide-by-small number problems.
    (Rns1,Rns2) = calc_normstokes(Rs0, Rs1, Rs2)
    (Gns1,Gns2) = calc_normstokes(Gs0, Gs1, Gs2)
    (Bns1,Bns2) = calc_normstokes(Bs0, Bs1, Bs2)

    (Nx,Ny) = Rs0.shape
    rgb_s0 = zeros((Nx,Ny,3), 'float32')
    rgb_s0[:,:,0] = Rs0
    rgb_s0[:,:,1] = Gs0
    rgb_s0[:,:,2] = Bs0

    rgb_ns1 = zeros((Nx,Ny,3), 'float32')
    rgb_ns1[:,:,0] = Rns1
    rgb_ns1[:,:,1] = Gns1
    rgb_ns1[:,:,2] = Bns1

    rgb_ns2 = zeros((Nx,Ny,3), 'float32')
    rgb_ns2[:,:,0] = Rns2
    rgb_ns2[:,:,1] = Gns2
    rgb_ns2[:,:,2] = Bns2

    return(rgb_s0, rgb_ns1, rgb_ns2)

## ============================================================
def draw_rgbpol_fft_circles():
    ## Plot the circles in a frequency domain of Nyquist units
    radius = 0.25
    c00   = [   0,   0]
    c10   = [ 0.5,   0]
    cm10  = [-0.5,   0]
    c01   = [   0, 0.5]
    c0m1  = [   0,-0.5]
    c11   = [ 0.5, 0.5]
    cm1m1 = [-0.5,-0.5]
    cm11  = [-0.5, 0.5]
    c1m1  = [ 0.5,-0.5]
    c20   = [   1,   0]
    c21   = [   1, 0.5]
    c2m1  = [   1,-0.5]
    cm20  = [  -1,   0]
    cm21  = [  -1, 0.5]
    cm2m1 = [  -1,-0.5]
    c02   = [   0,   1]
    c12   = [ 0.5,   1]
    cm12  = [-0.5,   1]
    c0m2  = [   0,  -1]
    c1m2  = [ 0.5,  -1]
    cm1m2 = [-0.5,  -1]

    centers1 = [c00,c10,cm10,c01,c0m1,c11,cm1m1,cm11,c1m1]
    centers2 = [c20,c21,c2m1]
    centers3 = [cm20,cm21,cm2m1]
    centers4 = [c02,c12,cm12]
    centers5 = [c0m2,c1m2,cm1m2]

    thetas = linspace(0.0, 2.0*pi, 200)

    for i in range(len(centers1)):
        xcircle = centers1[i][1] + radius * cos(thetas)
        ycircle = centers1[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, 'm-', lw=3)

    thetas = linspace(pi, 2.0*pi, 200)

    for i in range(len(centers2)):
        xcircle = centers2[i][1] + radius * cos(thetas)
        ycircle = centers2[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, 'b-', lw=3)

    thetas = linspace(0.0, pi, 200)

    for i in range(len(centers3)):
        xcircle = centers3[i][1] + radius * cos(thetas)
        ycircle = centers3[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, 'b-', lw=3)

    thetas = linspace(pi/2.0, 3.0*pi/2.0, 200)

    for i in range(len(centers4)):
        xcircle = centers4[i][1] + radius * cos(thetas)
        ycircle = centers4[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, 'r-', lw=3)

    thetas = linspace(-pi/2.0, pi/2.0, 200)

    for i in range(len(centers5)):
        xcircle = centers5[i][1] + radius * cos(thetas)
        ycircle = centers5[i][0] + radius * sin(thetas)
        plt.plot(xcircle, ycircle, 'r-', lw=3)

    return

