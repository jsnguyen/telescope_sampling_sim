from pathlib import Path
import subprocess
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import ndimage

from matplotlib_boilerplate import boilerplate
boilerplate.mpl_set_default_font('/Users/jsn/landing/docs/fonts/IBMPlexMono-Regular.ttf')
boilerplate.overwrite_mpl_defaults()

def make_mp4(frame_save_path, video_save_path, name):
    subprocess.run(['ffmpeg', '-y', '-r', '16', '-i', f'{frame_save_path}/{name}_%04d.png', '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', f'./{video_save_path}/{name}.mp4'])

def gaussian_2d(coords, center, amplitude, sigmas, offset):
    x,y = coords
    x = x + 0.5
    y = y + 0.5
    cx, cy = center
    sigma_x, sigma_y = sigmas
        
    return amplitude*np.exp(-(np.square((x-cx)/sigma_x) + np.square((y-cy)/sigma_y))/2) + offset

def orbit_distance(theta, semi_major, ecc):
    return (semi_major*(1-ecc*ecc))/(1+ecc*np.cos(theta))

def cartesian_to_polar(x, y):
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def keplers_equation(t, ecc, semi_major, period):

    mean_motion = 2*np.pi / period
    mean_anomaly = mean_motion*t
    kepler = lambda eccentric_anomaly: eccentric_anomaly-ecc*np.sin(eccentric_anomaly) - mean_anomaly 

    ecc_anom = fsolve(kepler, 0)

    true_anom = 2*np.arctan2(np.sqrt((1+ecc)/(1-ecc)),1/np.tan(ecc_anom/2))

    r = semi_major*(1-ecc*np.cos(ecc_anom))

    return r, true_anom

def plot_sampled_system(args):
    i, px, py, t, save_path, name, n_total_frames = args

    # sampled image params
    angular_size = 10 # arcsec
    angular_scale = 1 # au/arcsec
    image_res = 256 # pixels
    scale = image_res / angular_size #px/au
    image_shape = (image_res, image_res)
    image = np.zeros(image_shape)
    ys, xs = np.meshgrid(np.arange(image_res), np.arange(image_res))

    actual_px = scale*(px) + image_res/2
    actual_py = scale*(py) + image_res/2

    # star
    sigma_star = image_res * 0.05
    image += gaussian_2d((xs,ys), (image_res//2, image_res//2), 3, (sigma_star, sigma_star), 0)

    #planet
    sigma_planet = image_res * 0.025
    image += gaussian_2d((xs,ys), (actual_px, actual_py), 1, (sigma_planet, sigma_planet), 0)

    image = ndimage.rotate(image, 42, reshape=False)

    fig, ax = plt.subplots(figsize=(4,4))

    boxsize = angular_size*angular_scale
    ax.imshow(image, vmin=0, vmax=3, extent=[-boxsize, boxsize, -boxsize, boxsize], cmap='gray')
    #ax.imshow(image)
    ax.plot(0,0, color='black', marker='+')
    
    ax.set_title(f't = {t:.1f} days')
    ax.set_xlabel('[AU]')
    ax.set_ylabel('[AU]')

    save_name = f'{name}_{i:04}.png'
    print(f'Saving frame {i+1} of {n_total_frames}')
    plt.savefig(save_path/save_name, bbox_inches='tight')
    plt.close()

def plot_true_system(args):
    i, px, py, save_path, name, n_total_frames = args
    fig, ax = plt.subplots(figsize=(4,4))

    ax.plot(px,py, marker='o')
    ax.plot(0,0, marker='*')
    
    boxsize = 2
    ax.set_xlim(-boxsize, boxsize)
    ax.set_ylim(-boxsize, boxsize)
    
    ax.set_xlabel('[AU]')
    ax.set_ylabel('[AU]')

    save_name = f'{name}_{i:04}.png'
    print(f'Saving frame {i+1} of {n_total_frames}')
    plt.savefig(save_path/save_name, bbox_inches='tight')
    plt.close()

def main():

    # orbit params
    ecc = 0.8
    semi_major = 2
    m1 = 1 # solar mass
    m2 = 3.003e-6 # earth mass in solar mass
    G = 3.959e-4 # au, solar mass, days
    period = 2*np.pi * np.sqrt(np.power(semi_major,3)/(G*(m1+m2)))

    times = np.linspace(0, period, 120)
    rs = []
    thetas = []
    for t in times:
        r, theta = keplers_equation(t, ecc, semi_major, period)
        rs.append(r)
        thetas.append(theta)

    x,y = polar_to_cartesian(rs, thetas)

    name = 'frame'
    save_path = Path('./frames/')
    save_path.mkdir(parents=True, exist_ok=True)
    for fn in sorted(save_path.glob('*.png')):
        fn.unlink()

    args = []
    for i,(px,py,t) in enumerate(zip(x,y, times)):
        args.append((i, px, py, t, save_path, name, len(thetas)))

    pool = Pool()
    #pool.map(plot_true_system, args)
    pool.map(plot_sampled_system, args)

    make_mp4(save_path, '.', name)

if __name__=='__main__':
    main()