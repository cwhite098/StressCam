# import the necessary packages
from scipy.spatial import ConvexHull
import numpy as np


def get_hull(pts):
    # Function that gets the hull (outline) of a set of points
    hull = ConvexHull(pts)
    # Returns those points that form the outline of the shape, in the correct order.
    return pts[hull.vertices]