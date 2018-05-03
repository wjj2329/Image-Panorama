#!/usr/bin/env python3
import os
import argparse
from scipy import misc
import scipy.sparse.csgraph as csgraph
import scipy.sparse.csr as csr
import math
import cv2
import numpy as np
inf=float('inf')
neginf=float("-inf")
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher_create(cv2.NORM_L2)

#gets the fitting rectangle for it
def get_my_rect(*points):
    top = inf
    left = inf
    right = neginf
    bottom = neginf
    for x, y in points:
        if x < left:
            left = x
        if x > right:
            right = x
        if y < top:
            top = y
        if y > bottom:
            bottom = y
    left = int(math.floor(left))
    top = int(math.floor(top))
    height = int(math.ceil(bottom - top))
    width = int(math.ceil(right - left))
    return (left, top), (width, height)

#this is where I append them together
def insert_image(base, img, shift):
    height, width = img.shape[:2]
    x, y = shift
    insert_loc = np.s_[y:y + height, x:x + width]
    dest = base[insert_loc]
    mask = (255 - img[:,:, 3])
    dest_background = cv2.bitwise_and(dest, dest, mask=mask)
    dest = cv2.add(dest_background, img)
    base[insert_loc] = dest

#gets the corners of the picture
def image_corners(pic):
    return np.array([ [0., 0.],[0., pic.shape[1]],pic.shape[:2],[pic.shape[0], 0.],])

class Image:

    def __init__(self, image, name):
        self.image = image
        self.name = name
        self.kp = None
        self.des = None
        self.kp, self.des = sift.detectAndCompute(self.image, None)

class Stitcher:
    def __init__(self):
        self.matches = {}
        self.image_list = []
        self.center_img = None
        self.connect_matrix = None

    def add_image(self, image, name: str=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image = Image(image, name=name)
        index = len(self.image_list)
        self.image_list.append(image)
        for otherindex, other in enumerate(self.image_list[:-1]):
            match = self.match_des(image, other)
            self.matches[(index, otherindex)] = match

    @property
    def center(self):
        if self.center_img is None:
            self.center_img = self.find_center_image()
        return self.center_img

    @center.setter
    def center(self, val):
        self.center_img = val

    def edges(self, new_corners):
        all_corners = []
        for corners in new_corners:
            all_corners.extend(corners)
        corner, size = get_my_rect(*all_corners)
        return corner, size

    def stitch(self):
        parents = csgraph.dijkstra(self.edge_matrix,directed=False, indices=self.center,return_predecessors=True,)[1]
        next_to_go = self.get_rel_homographies(parents)
        current = self.get_homographies(parents, next_to_go)
        my_corners = self.get_my_corners(current)
        base_shift, base_size = np.array(self.edges(my_corners))
        order = self.order_to_stitch(parents)
        canvas = np.zeros((base_size[1], base_size[0], 4), dtype=np.uint8)
        for x in order:
            image = self.image_list[x]
            new_corners = my_corners[x]
            homg = current[x]
            shift, size = np.array(get_my_rect(*new_corners))
            dest_shift = shift - base_shift
            trans = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]]) # the shift that we talked about in class to get it in the right place
            homgtrans = trans.dot(homg)
            new_image = cv2.warpPerspective(image.image, homgtrans, tuple(size),flags=cv2.INTER_LINEAR,)
            insert_image(canvas, new_image, dest_shift)
        return canvas

    def order_to_stitch(self, parents):
          return csgraph.depth_first_order(csgraph.reconstruct_path(self.edge_matrix, parents, directed=False),self.center,return_predecessors=False,)[::-1]

    def get_my_corners(self, current):
        my_corners = []
        for image, H in zip(self.image_list, current):
            corners = image_corners(image.image)
            new_corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2), H)
            new_corners = new_corners[0]
            my_corners.append(new_corners)
        return my_corners

    def get_match(self, src_index, dst_index):
        if (src_index, dst_index) in self.matches:
            return self.matches[(src_index, dst_index)]
        return self.matches[(dst_index, src_index)]

    def get_homographies(self, parents, next_to_go):
        c = self.center
        my_homographies = [None] * len(parents)
        my_homographies[c] = next_to_go[c]
        path = []
        while any(i is None for i in my_homographies):
            path.append(next(n for n, i in enumerate(my_homographies) if i is None))
            while path:
                src_index = path.pop()
                dst_index = parents[src_index]
                if c == src_index:
                    continue
                if my_homographies[dst_index] is None:
                    path.extend((src_index, dst_index))
                else:
                    my_homographies[src_index] = next_to_go[src_index].dot(my_homographies[dst_index])
        return my_homographies

    def get_rel_homographies(self, parents):
        c = self.center
        next_to_go = []
        for src_index, dst_index in enumerate(parents):
            if dst_index < 0 or src_index == c:
                next_to_go.append(np.identity(3))
                continue
            matches = self.get_match(src_index, dst_index)
            swap = (src_index, dst_index) not in self.matches
            src, dst = self.image_list[src_index], self.image_list[dst_index]
            temp = self.get_homography(src, dst, matches, swap=swap)
            next_to_go.append(temp)
        return next_to_go

    def get_homography(self,src,dst,matches,swap=False):
        if swap:
            src, dst = dst, src
        src_data = np.array([src.kp[i.queryIdx].pt for i in matches],dtype=np.float64).reshape(-1, 1, 2)
        dst_data = np.array([dst.kp[i.trainIdx].pt for i in matches],dtype=np.float64).reshape(-1, 1, 2)
        if swap:
            src_data, dst_data = dst_data, src_data
            src, dst = dst, src
        temp = cv2.findHomography(src_data, dst_data, cv2.RANSAC, 2.)
        return temp[0]

    def find_center_image(self):
        shortest_path = csgraph.shortest_path(self.edge_matrix, directed=False,)
        center = np.argmin(shortest_path.max(axis=1))
        return center

    @property
    def edge_matrix(self):
        if len(self.image_list) == 0:
            raise ValueError('I must have at least one image to do this!')
        current = self.connect_matrix
        if current is not None and current.shape[0] == len(self.image_list):
            return current
        mymatches = list(self.matches)
        base = max(len(v) for v in self.matches.values()) + 1
        values = [base - len(self.matches[i]) for i in mymatches]
        self.connect_matrix = csr.csr_matrix((values, tuple(np.array(mymatches).T)),shape=(len(self.image_list), len(self.image_list)),)
        return self.connect_matrix

    def match_des(self, src, dst):
        matches = bf.knnMatch(src.des, dst.des, k=2)
        # Ratio test it seemed that .75 was the best!
        return [i for i, j in matches if i.distance < 0.75 * j.distance]
class solver_class():

  def calc_Pic(self, pics_to_be_stitched, output, base ):
    stitch = Stitcher()
    stitch.center = base
    for image in pics_to_be_stitched:
        stitch.add_image(image)
    final_pic = stitch.stitch()
    cv2.imwrite(output, cv2.cvtColor(final_pic, cv2.COLOR_RGBA2BGRA))
