# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:43:22 2022

@author: Sylvain
"""
import numpy as np
from PIL import Image
from piles import PILe, ImageDraws
from PIL import Image, ImageDraw

import scipy.spatial
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from matplotlib.collections import PolyCollection

#%% Shared functions
def check_img_hmap(img, heightmap=None):
    
    if not isinstance(img, Image.Image):
        raise TypeError('Image must be a valid PIL image')
       
    img = img.convert('RGBA')
    if heightmap == None:
        heightmap = Image.new('L', img.size, color = (255))
    
    else:
        heightmap = heightmap.convert('L')
        if heightmap.size != img.size:
            heightmap = heightmap.resize(img.size)
    
    heightmap = np.array(heightmap) / 255.
    
    return img, heightmap

def RGB_std(arr, hmap):
    
    valid_idx = np.where(arr[:, :, 3] != 0)
    arr = arr[valid_idx]
    hmap = hmap[valid_idx]
    
    arr = arr[:, :-1]

    std = np.max(np.std(arr, axis=0))

    col = np.mean(arr, axis=0).astype(np.uint8)
    col = tuple(col)
    
    hmap_weight = np.max(hmap)
    
    return std, col, hmap_weight

#%% Dithering
def dither(img, kernel='Floyd-Steinberg', nc=2):

    def _get_new_val(old_val, nc):
        """
        Get the "closest" colour to old_val in the range [0,1] per channel divided
        into nc values. This works well for B&W pictures, but nor for RGB ones.
        If nc = 2, this means 2 possible values per channel and hence 2^3 = 8 different colors.

        """

        return np.round(old_val * (nc - 1)) / (nc - 1)

    dither_kernels = {'Floyd-Steinberg': [[0, 0, 0],
                                          [0, 0, 7],
                                          [3, 5, 1]],

                      'Jarvis-Judis-Ninke': [[0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 7, 5, 0],
                                             [0, 0, 3, 5, 7, 5, 3],
                                             [0, 0, 1, 3, 5, 3, 1]],

                      'Stucki': [[0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 8, 4, 0],
                                 [0, 0, 2, 4, 8, 4, 2],
                                 [0, 0, 1, 2, 4, 2, 1]],

                      'Atkinson': [[0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1],
                                   [0, 1, 1, 1, 0],
                                   [0, 0, 1, 0, 0]],

                      }
    if kernel not in dither_kernels.keys():
        raise ValueError(
            f'Available dithering kernels are {dither_kernels.keys()}')

    width, height = img.size
    arr = np.array(img, dtype=float) / 255

    ker = dither_kernels[kernel]
    ker = ker / np.sum(ker)

    ker_h, ker_w, = ker.shape

    ker_h = ker_h // 2
    ker_w = ker_w // 2

    pad = max(ker_h, ker_w)

    if len(arr.shape) == 3:

        arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        ker = np.repeat(ker[:, :, np.newaxis], 3, axis=2)

    else:
        arr = np.pad(arr, pad)

    for ir in range(height):
        for ic in range(width):

            old_val = arr[ir+pad, ic+pad].copy()
            new_val = _get_new_val(old_val, nc)

            arr[ir+pad, ic+pad] = new_val
            err = old_val - new_val
            err_ker = err * ker

            arr[ir + pad - ker_h: ir + pad + ker_h + 1,
                ic + pad - ker_w: ic + pad + ker_w + 1] += err_ker

    carr = np.array(arr/np.max(arr, axis=(0, 1)) * 255,
                    dtype=np.uint8)[pad:-pad, pad:-pad]

    dithered = Image.fromarray(carr)

    return dithered

#%% Quadtrees
def quadtree(img, std_thr, heightmap=None, level=0, max_level=6):
    
    img, heightmap = check_img_hmap(img, heightmap)

    # We will save the coordinates in this dict.
    results = {'top': [],
               'left': [],
               'x': [],
               'y': [],
               'width': [],
               'height': [],
               'color': [],
               'poly': [],
               'level': []}

    
    def subdivide(arr, thr, topleft, widthheight, results, heightmap, level=0, max_level=max_level):

        left, top = topleft  # not smart...
        width, height = widthheight
        to_check = arr[int(top):int(top+height), int(left):int(left+width)]
        hmap_to_check = heightmap[int(top):int(top+height), int(left):int(left+width)]

        std, col, hmap_weight = RGB_std(to_check, hmap_to_check)
                
        if np.isnan(std):
            raise ValueError(
                'Error happened at coordinates x={left}, y={top} with width={width}, height={height}')

        # Ending if std below threshold or if the subdivision is 1 pixel
        if ((std < thr) | (level >= max_level) | (hmap_weight - (level*0.1) < 0.1)):
            # And saving the values, of course.
            results['top'] += [top]
            results['left'] += [left]
            results['x'] += [left + width/2]
            results['y'] += [top + height/2]
            results['width'] += [width]
            results['height'] += [height]
            results['color'] += [col]
            results['level'] += [level]
            
            poly = [[left, top], [left+width, top], 
                    [left+width, top+height], [left, top+height]]
            results['poly'] +=  [np.asarray(poly)]

            return

        else:

            x2 = left + width/2
            y2 = top + height/2

            # Coordinates of the 4 top-left corner of the new subdivisions.
            c1 = (left, top)
            c2 = (x2, top)
            c3 = (x2, y2)
            c4 = (left, y2)
            # And their new dimensions.
            new_width = width / 2
            new_height = height / 2

            # Aaaand recursion.
            for c in [c1, c2, c3, c4]:
                subdivide(arr, thr, c, (new_width, new_height), results, heightmap, level=level+1, max_level=max_level)

    img_width, img_height = img.size

    arr = np.asarray(img)

    subdivide(arr, std_thr, (0, 0), (img_width, img_height), results, heightmap, level=0, max_level=max_level)

    return results

#%% Voronoitrees
def voronoitree(img, npoints=10, max_level=6, std_thr=40, heightmap=None):
    
    img, heightmap = check_img_hmap(img, heightmap)
        
    
    def bounded_voronoi(points):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        Code by Pauli Virtanen, see https://gist.github.com/pv/8036995
        """
    
        vor = scipy.spatial.Voronoi(points)
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        radius = vor.points.ptp().max() * 2
    
        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
    
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
    
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue
                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
    
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
    
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
    
            # finish
            new_regions.append(new_region.tolist())
        return new_regions, np.asarray(new_vertices)
    
    
    def poly_random_points_safe(V, n=10):
        """ Random points inside a convex polygon (guaranteed)
        V : numpy array
            Polygon border
        n : int
            Number of points to sample
        """
    
        def random_point_inside_triangle(A, B, C):
            r1 = np.sqrt(np.random.uniform(0, 1))
            r2 = np.random.uniform(0, 1)
            return (1 - r1) * A + r1 * (1 - r2) * B + r1 * r2 * C
    
        def triangle_area(A, B, C):
            return 0.5 * np.abs(
                (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
            )
    
        # Cheap trianglulation of the polygon
        C = V.mean(axis=0)
        T = [(C, V[i], V[i + 1]) for i in range(len(V) - 1)]
        A = np.array([triangle_area(*t) for t in T])
        A /= A.sum()
    
        points = [C]
        for i in np.random.choice(len(A), size=n - 1, p=A):
            points.append(random_point_inside_triangle(*T[i]))
        return points
    
    
    def extract_polygon(img, hmap, poly):
        coords = list(poly.exterior.coords)
        xs = [a for (a, b) in coords]
        ys = [b for (a, b) in coords]
    
        left = int(np.min(np.floor(xs)))
        top = int(np.min(np.floor(ys)))
        right = int(np.max(np.ceil(xs)))
        bottom = int(np.max(np.ceil(ys)))
    
        cropped_img = img.crop((left, top, right, bottom))
        cropped_hmap = hmap[top:bottom, left:right]
    
        new_width, new_height = cropped_img.size
    
        try:
            cropped_img = np.array(cropped_img)
        except SystemError: # happens with rounding errors and array with 0-dimensions
            print(left, top, right, bottom)
            return
        
        shifted_poly = list(tuple(zip(xs - np.min(xs), ys - np.min(ys))))
    
        mask_Im = Image.new("RGBA", (cropped_img.shape[1], cropped_img.shape[0]), 0)
        tmp_drawer = ImageDraw.Draw(mask_Im)
        tmp_drawer.polygon(
            shifted_poly, fill=(255, 255, 255, 255), outline=(255, 0, 0, 255), width=1
        )
    
        mask = np.array(mask_Im)
    
        cropped_img[:, :, 3] = mask[:, :, 3]
    
        centre_x = np.min(xs) + (np.max(xs) - np.min(xs)) / 2
        centre_y = np.min(ys) + (np.max(ys) - np.min(ys)) / 2
    
        r = {
            "top": [top],
            "left": [left],
            "x": [centre_x],
            "y": [centre_y],
            "width": [width],
            "height": [height],
            "shifted_polys": [shifted_poly],
        }
    
        return cropped_img, cropped_hmap, r
    
       
    
    def voronoi(img, V, results, npoints, level, max_level, std_thr, heightmap):
        """ Recursive voronoi """
    
        points = poly_random_points_safe(V, npoints - level)
        regions, vertices = bounded_voronoi(points)
        clip = Polygon(V)
    
        clipped_img, clipped_hmap, temp_results = extract_polygon(img, heightmap, clip)
    
        std, col, hmap_weight = RGB_std(clipped_img, clipped_hmap)
    
        if (level == max_level) | (std < std_thr) | (hmap_weight - (level*0.1) < 0.1):
    
            tile = Image.fromarray(clipped_img, "RGBA")
    
            for k, v in temp_results.items():
                results[k] += temp_results[k]
            results["images"] += [tile]
            results["color"] += [col]
            results["level"] += [level]
            results['poly'] += [np.array([point for point in clip.exterior.coords])]
    
            
    
            return
    
        for region in regions:
            polygon = Polygon(vertices[region]).intersection(clip) # Using Shapely for the clipping 
            polygon = np.array([point for point in polygon.exterior.coords])
            voronoi(img, polygon, results, npoints, level + 1, max_level, std_thr, heightmap)
            
        return 
    
    
    results = {
        "top": [],
        "left": [],
        "x": [],
        "y": [],
        "width": [],
        "height": [],
        "color": [],
        "poly": [],
        "shifted_polys" : [],
        "images": [],
        "level": []
    }
    
    
    
    width, height = img.size
    
    first_poly = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    
    voronoi(img, first_poly, results, npoints, level=0, max_level=max_level, std_thr=std_thr, heightmap=heightmap)
    
    return results


#%% Function call
link = "pigeon.jpg"
link_hmap = "pigeon_map2.jpg"

img = Image.open(link).convert("RGBA")
hmap = Image.open(link_hmap).convert("L")

width, height = img.size

poly1 = quadtree(img, std_thr=0, heightmap=hmap, max_level=10)
#%%
poly2 = voronoitree(img, std_thr=0, heightmap=hmap, max_level=6)

#%%
fig, axes = plt.subplots(figsize=(16,8), ncols=2, nrows=1)

ax = axes.ravel()

for i, res in enumerate([poly1, poly2]):
    
    ax[i].set_xlim(0, width)
    ax[i].set_ylim(0, height)
    ax[i].set_aspect('equal')
    ax[i].axis("off")
    ax[i].invert_yaxis()
    
    n_polys = len(res['poly'])
    mpl_colors = [(a/255, b/255, c/255) for (a, b, c) in res['color']]
        
    collection = PolyCollection(res['poly'],
                                linewidth = 1, 
                                facecolor = mpl_colors,
                                edgecolor = mpl_colors
    )
    
    ax[i].add_collection(collection)

fig.tight_layout()
# fig.savefig('recursive-voronoi.pdf')
# fig.savefig("recursive-voronoi.png", bbox_inches='tight', dpi=123, pad_inches = 0)
fig.show()