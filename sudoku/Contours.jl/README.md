# Contours.jl

## Definitions

This package provides basic functionality for working with contours in images. 

This package uses formulas derived for polygons, where a "polygon" is a separate concept to a "contour".
It is important to distinguish between the two:

> `contour::Array{CartesianIndex}` A linked chain of connected discrete points $(i, j)$. Each point in the chain differs by magnitude 1 from the previous point. 

> `vertices::AbstractArray` A set of points $(x, y)$ which define the vertices of edges of a polygon, simple or complex. The co-ordinates should be real numbers. Otherwise there is no restriction on their values. 

A contour is a subtype of a polygon. Therefore algorithms which are valid for polygons are valid for contours.

# Algorithms

- `find_contours` Contour finding algorithm in a binary image. The algorithm is from "Topological 
Structural Analysis of Digitized Binary Images by Border Following" by Suzuki and Abe (Same as OpenCV) 
- `point_in_polygon` A very robust ray-tracing point in polygon algorithm which handles edge cases well. The algorithm is from "A Simple and Correct Even-Odd Algorithm for the Point-in-Polygon Problem for Complex Polygons" 
by Michael Galetzka and Patrick Glauner (2017).
- Moments of the polygon: 
    - `centroid_contour`, `centroid_polygon`. These functions are identical.
    - `area_contour`, `area_polgyon`. These functions are identical. For a discrete area rather use the `fill_contour!` function on a blank grid.
- Drawing functions:
    - contours only: `draw_contour!`, `draw_contours!`
    - filled contours: `fill_contour!`, `fill_contours!`, `AbstractFillAlgorithm`, `Boundary4Fill`, `ScanFill`. Using `Boundary4Fill` is much faster than `ScanFill` but it will most likely miss values.
