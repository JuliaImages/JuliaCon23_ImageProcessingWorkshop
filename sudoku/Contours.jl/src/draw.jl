# a contour is a vector of 2 int arrays
"""
    draw_contour!(image, color, contour)

Draw the boundary of a contour.
"""
function draw_contour!(image::AbstractArray, contour::Array{CartesianIndex}, color)
    for ind in contour
        image[ind] = color
    end
end

"""
    draw_contours!(image, color, contours)

Draw the boundary of an array of contours.
"""
function draw_contours!(image::AbstractArray, contours::AbstractArray, color)
    for cnt in contours
        draw_contour!(image, cnt, color)
    end
end



abstract type AbstractFillAlgorithm end

"""
    fill_contours!(image, color, contours, [f::AbstractFillAlgorithm])

For f, use:
1. Boundary4Fill() - fast but not robust
2. ScanFill() - slower but more robust. Default
"""
function fill_contours!(image::AbstractArray, contours,  color, f::AbstractFillAlgorithm)
    for cnt in contours
        fill_contour!(image, cnt, color, f)
    end
end
fill_contours!(image::AbstractArray, contours,  color) = fill_contours!(image, contours, color, ScanFill())

"""
    fill_contour!(image, color, contour, [f::AbstractFillAlgorithm])

For f, use:
1. Boundary4Fill() - fast but not robust
2. ScanFill() - slower but more robust. Default
"""
function fill_contour!(image::AbstractArray, contour::Array{CartesianIndex}, color, f::AbstractFillAlgorithm)
    f(image, contour, color)
end
fill_contour!(image::AbstractArray, contour::Array{CartesianIndex}, color) = fill_contour!(image, contour, color, ScanFill())

function get_seed(contour)
    seed = contour[1]
    for candidate in contour
        candidate += CartesianIndex(0, +1)
        if point_in_polygon(candidate, contour, false)
            seed = candidate
            break
        end
    end 
    return seed
end

"""
    Boundary4Fill(image, contour, color)

Fast but not robust.
Flood fill from an intial seed until get to boundaries.
"""
struct Boundary4Fill <: AbstractFillAlgorithm 
end

function (f::Boundary4Fill)(image::AbstractArray, contour::Array{CartesianIndex}, color)
    height, width = size(image)
    visited = falses(height, width)
    seed = get_seed(contour)
    top, bottom, left, right = seed[1], seed[1], seed[2], seed[2]
    for ind in contour
        visited[ind] = true
        image[ind] = color
        top = min(top, ind[1])
        bottom = max(bottom, ind[1])
        left = min(left, ind[2])
        right = max(right, ind[2])
    end
    bounds = [top, bottom, left, right]
    stack = [seed]
    while length(stack) > 0
        ind = pop!(stack)
        if visited[ind]
            continue
        end
        image[ind] = color
        visited[ind] = true
        i, j = ind[1], ind[2]
        for neighbour in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
            neighbour = CartesianIndex(neighbour)
            if in_bounds(bounds, neighbour) && !visited[neighbour]
                push!(stack, neighbour)
            end
        end
    end
end

function in_bounds(bounds, point)
    top, bottom, left, right = bounds
    (top <= point[1] <= bottom) && (left <= point[2] <= right)
end


"""
    ScanFill(image, contour, color)

Slow but robust.
Scan row by row, column by column. After a vertix is crossed, check if the point is inside the contour.
"""
struct ScanFill <: AbstractFillAlgorithm 
end

function (f::ScanFill)(image::AbstractArray, contour::Array{CartesianIndex}, color)
    #= 
    The simple algorithm is to check every point in the loop.
    The following optimisations are applied:
    1. Only check point_in_polygon after crossing a boundary.
    2. If pixel above is inside polygon, skip point_in_polygon check.
    3. Only pass subset of vertices to point_in_polygon.
    4. For the subset of vertices, keep boolean array of active vertix indices 
    =#
    seed = first(contour)
    top, bottom, left, right = seed[1], seed[1], seed[2], seed[2]
    for ind in contour
        top = min(top, ind[1])
        bottom = max(bottom, ind[1])
        left = min(left, ind[2])
        right = max(right, ind[2])
    end

    labels = [ind[1] for ind in contour]

    inside = false
    crossed = false
    inner = falses(size(image)...)
    for i in top:bottom
        active = (labels .== (i - 1)) .| (labels .== i) .| (labels .== (i + 1)) 
        edges = contour[active]
        for j in left:right 
            ind = CartesianIndex(i, j)
            if ind in edges
                inside = false
                crossed = true
                image[ind] = color
            elseif !inside && crossed
                inside = (i > 2 && inner[i - 1, j]) || point_in_polygon(ind, edges)
                crossed = false
            end
            if inside
                image[ind] = color
                inner[ind] = true
            end
        end
    end
end
