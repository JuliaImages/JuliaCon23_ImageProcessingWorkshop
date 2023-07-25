
include("Contours.jl/src/Contours.jl");
using .Contours

struct ConnectedComponentStatistics
    left::Int
    top::Int
    right::Int
    bottom::Int
    area::Int
    centroid::Tuple{Float64, Float64}
end

"""
Get statistics for the output of Images.label_components
"""
function calc_connected_component_statistics(labels::AbstractArray, label::Int)
    height, width = size(labels)

    left = width
    top = height
    right = 0
    bottom = 0
    area = 0
    Cx, Cy = 0.0, 0.0

    for i in 1:height
        for j in 1:width
            if labels[i,j] == label
                area += 1
                left = min(left, j)
                top = min(top, i)
                right = max(right, j)
                bottom = max(bottom, i)
                Cx += 1.0
                Cy += 1.0
            end
        end
    end
    ConnectedComponentStatistics(left, top, right, bottom, area, (Cx/area, Cy/area))
end

function read_digits(
    image::AbstractArray,
    model; 
    offset_ratio=0.1,
    radius_ratio::Float64=0.25, 
    detection_threshold::Float64=0.10, 
    )
    height, width = size(image)
    step_i = ceil(Int, height / 9)
    step_j = ceil(Int, width / 9)
    offset_i = round(Int, offset_ratio * step_i)
    offset_j = round(Int, offset_ratio * step_j)

    grid = zeros(Int, (9, 9))
    centres =  [(-1.0, -1.0) for i in 1:9, j in 1:9]
    probabilities = zeros(Float32, (9, 9))

    for (i_grid, i_img) in enumerate(1:step_i:height)
        for (j_grid, j_img) in enumerate(1:step_j:width)
            prev_i = max(1, i_img - offset_i)
            prev_j = max(1, j_img - offset_j)
            next_i = min(i_img + step_i + offset_i, height)
            next_j = min(j_img + step_j + offset_j, width)
            RoI = image[prev_i:next_i, prev_j:next_j]
            if detect_in_centre(RoI)
                centre, digit = extract_digit(RoI, radius_ratio=radius_ratio, threshold=detection_threshold)
                ŷ, prob = prediction(model, digit)
                grid[i_grid, j_grid] = ŷ
                
                centre = (centre[1] + prev_i, centre[2] + prev_j)
                probabilities[i_grid, j_grid] = prob
            else
                centre = (prev_i + step_i/2, prev_j + step_j/2)
            end
            centres[i_grid, j_grid] = centre
        end
    end
    grid, centres, probabilities
end


function extract_digit(image_in::AbstractArray; kwargs...)
    image = copy(image_in)
    # have to binarize again because of warping
    image = binarize(image, Otsu()) # global binarization algorithm

    labels = label_components(image) 

    height, width = size(image)
    for i in 1:length(unique(labels))
        image_label = copy(image)
        image_label[labels .!= i] .= 0
        if detect_in_centre(image_label; kwargs...)
            stats = calc_connected_component_statistics(labels, i)
            width_label = abs(stats.right - stats.left)
            height_label = abs(stats.bottom - stats.top)
            length_  = max(width_label, height_label)

            # note: the centroid is not a good chocie for a visual centre 
            centre = (stats.top + Int(round(height_label/2)), stats.left + Int(round(width_label/2)))

            # make square
            top = max(1, floor(Int, centre[1] - length_/2))
            left = max(1, floor(Int,centre[2] - length_/2))
            bottom = min(height, ceil(Int, centre[1] + length_/2))
            right = min(width, ceil(Int, centre[2] + length_/2))
            return centre, image_label[top:bottom, left:right]
        end
    end
    (height/2, width/2), image
end


"""
detect_in_centre(image::AbstractArray; [radius_ratio], [threshold])

Detect an object in a region of interest. This is done by convolving it with a circle.
"""
function detect_in_centre(image::AbstractArray; radius_ratio::Float64=0.25, threshold::Float64=0.10)
    height, width = size(image)
    radius = min(height, width) * radius_ratio
    kernel = make_circle_kernel(height, width, radius)
    conv = kernel .* image
    detected = sum(conv .!= 0)/(pi * radius * radius) > threshold
    detected
end


function make_circle_kernel(height::Int, width::Int, radius::Float64)
    # backward algorithm
    kernel = zeros((height, width))
    centre = (width/2, height/2)
    for i in 1:height
        for j in 1: width
            z = radius^2 - (j - centre[1])^2 - (i - centre[2])^2
            if z > 0
                kernel[CartesianIndex(i, j)] = 1
            end
        end
    end
    kernel
end

make_circle_kernel(height::Int, width::Int, radius::Int) = make_circle_kernel(height, width, Float64(radius))


function pad_image(image::AbstractArray{T}; pad_ratio=0.15) where T
    height, width = size(image)
    pad = floor(Int, pad_ratio * max(height, width))
    imnew = zeros(T, (height + 2pad, width + 2pad))
    imnew[(pad + 1):(pad + height), (pad + 1):(pad + width)] = image
    imnew
end


function prediction(model, image::AbstractArray, pad_ratio=0.2)
    image = pad_image(image, pad_ratio=pad_ratio)
    image = imresize(image, (28, 28))
    x = Flux.batch([Flux.unsqueeze(Float32.(image), 3)])
    logits = model(x)
    probabilites = softmax(logits)
    idx = argmax(probabilites)
    ŷ = idx[1] - 1
    ŷ, probabilites[idx]
end

function align_centres(centres::Matrix, guides::BitMatrix)
    centres_aligned = copy(centres)
    if size(centres) != size(guides)
         throw("$(size(centres)) != $(size(guides)), sizes of centres and guides must be the same.")
    end
    for i in 1:size(centres, 1)
        for j in 1:size(centres, 2)
            if !guides[i, j]
                # y is common to row i
                if any(guides[i, :])
                    ys = [point[1] for point in centres[i, :]] .* guides[i, :]
                    Cy = sum(ys) / count(guides[i, :])
                else
                    Cy = centres[i, j][1]
                end
                #  x is common to column j
                if any(guides[:, j])
                    xs = [point[2] for point in centres[:, j]] .* guides[:, j]
                    Cx = sum(xs) / count(guides[:, j])
                else 
                    Cx = centres[i, j][2]
                end
                centres_aligned[i, j] = (Cy, Cx)
            end
        end
    end
    centres_aligned
end

function invert_image(image)
    image_inv = Gray.(image)
    height, width = size(image)
    for i in 1:height
        for j in 1:width
            image_inv[i, j] = 1 - image_inv[i, j]
        end
    end
    return image_inv
end

function fit_rectangle(points::AbstractVector)
    # return corners in top-left, top-right, bottom-right, bottom-left
    min_x, max_x, min_y, max_y = typemax(Int), typemin(Int), typemax(Int), typemin(Int)
    for point in points
        min_x = min(min_x, point[1])
        max_x = max(max_x, point[1])
        min_y = min(min_y, point[2])
        max_y = max(max_y, point[2])
    end
    
    corners = [
        CartesianIndex(min_x, min_y),
        CartesianIndex(max_x, min_y),
        CartesianIndex(max_x, max_y),
        CartesianIndex(min_x, max_y),
    ]

    return corners
end


function fit_quad(points::AbstractVector) 
    rect = fit_rectangle(points)

    corners = copy(rect)
    distances = [Inf, Inf, Inf, Inf]

    for point in points
        for i in 1:4
            d = abs(point[1] - rect[i][1]) + abs(point[2] - rect[i][2])
            if d < distances[i]
                corners[i] = point
                distances[i] = d
            end
        end
    end
    return corners
end


"""
get_perspective_matrix(source::AbstractArray, destination::AbstractArray)

Compute the elements of the matrix for projective transformation from source to destination.
Source and destination must have the same number of points.

Transformation is:
| u |   | c11 c12 c13 | | x |
| v | = | c21 c22 c23 |·| y |
| w |   | c31 c32   1 | | 1 |

So that u' = u/w, v'=v/w where w = 1/(focal length) of the pinhole camera. 

Sovling for u and v:
    (c31·x + c32·y + 1)u = c11·x + c12·y + c13
    (c31·x + c32·y + 1)v = c21·x + c22·y + c23
Similarly for v and rearrange into the following matrix:
    [u1; u2; u3; u4; v1; v2; v3; v4] = A·[c11; c12; c13; c21; c22; c23; c31; c32]
    B = A ⋅ X
"""
function get_perspective_matrix(source::AbstractArray, destination::AbstractArray)
    if (length(source) != length(destination))
        error("$(length(source))!=$(length(destination)). Source must have the same number of points as destination")
    elseif length(source) < 4
        error("length(source)=$(length(source)). Require at least 4 points")
    end
    indx, indy = 1, 2
    n = length(source)
    A = zeros(2n, 8)
    B = zeros(2n)
    for i in 1:n
        A[i, 1] = source[i][indx]
        A[i, 2] = source[i][indy]
        A[i, 3] = 1
        A[i, 7] = -source[i][indx] * destination[i][indx]
        A[i, 8] = -source[i][indy] * destination[i][indx]
        B[i] = destination[i][indx]
    end
    for i in 1:n
        A[i + n, 4] = source[i][indx]
        A[i + n, 5] = source[i][indy]
        A[i + n, 6] = 1
        A[i + n, 7] = -source[i][indx] * destination[i][indy]
        A[i + n, 8] = -source[i][indy] * destination[i][indy]
        B[i + n] = destination[i][indy]
    end
    M = inv(A) * B
    M = [
        M[1] M[2] M[3];
        M[4] M[5] M[6]'
        M[7] M[8] 1
    ]
    M
end


function order_points(corners)
	# order points: top-left, top-right, bottom-right, bottom-left
	rect = zeros(typeof(corners[1]), 4)
	# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
	s = [point[1] + point[2] for point in corners]
	rect[1] = corners[argmin(s)]
	rect[3] = corners[argmax(s)]
	# now, compute the difference between the points, the top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = [point[2] - point[1] for point in corners]
	rect[2] = corners[argmin(diff)]
	rect[4] = corners[argmax(diff)]
	# return the ordered coordinates
	return rect
end


function four_point_transform(image::AbstractArray, corners::AbstractVector)
    quad = order_points(corners)
    rect = fit_rectangle(corners)
    destination = [CartesianIndex(point[1] - rect[1][1] + 1, point[2] - rect[1][2] + 1) for point in rect]
    maxWidth = destination[2][1] - destination[1][1] 
    maxHeight = destination[3][2] - destination[2][2] 

    M = get_perspective_matrix(quad, destination)
    invM = inv(M)
    transform = perspective_transform(invM)

    warped = warp(image, transform, (1:maxWidth, 1:maxHeight))
    warped, invM
end

extend1(v) = [v[1], v[2], 1]
perspective_transform(M::Matrix) = PerspectiveMap() ∘ LinearMap(M) ∘ extend1


function rgb_to_float(pixel::AbstractRGB)
    Float32.([red(pixel), green(pixel), blue(pixel)])
end


function get_color(image::AbstractArray{T}, point) where T
    ind = CartesianIndex(Int(floor(point[1])), Int(floor(point[2])))
    image[ind]
end

"""
imwarp(image, invM, dest_size)

This function is only for illustrative purposes only.
It is a slower and less accurate version of ImageTransformations.warp.
It implements a backwards transformation algorithm for a homography transformation matrix. 
Colors are approximated as the pixel the inverse transform lands in.
No further interpolation is done.
"""
function imwarp(image::AbstractArray{T}, invM::Matrix, dest_size::Tuple{Int, Int}) where T
    warped = zeros(T, dest_size...)

    height, width = dest_size
    for i in 1:height
        for j in 1:width
            ind = apply_homography((i, j), invM)
            warped[CartesianIndex(i, j)] = get_color(image, ind)
        end
    end          
    warped
end

function detect_grid(image::AbstractArray; kwargs...)
    blackwhite = preprocess(image; kwargs...)

    # assumption: grid is the largest contour in the image
    contours = find_contours(blackwhite, external_only=true)
    idx_max = argmax(map(area_contour, contours))
    quad = fit_quad(contours[idx_max])
    
    blackwhite, quad
end


function preprocess(
    image::AbstractArray; 
    max_size=1024, 
    blur_window_size=5, σ=1, 
    threshold_window_size=15, threshold_percentage=7
    )
    gray = Gray.(image)

    # resize
    ratio = max_size/size(gray, argmax(size(gray)))
    if ratio < 1
        gray = imresize(gray, ratio=ratio)
    end
    
    # blur
    kernel = Kernel.gaussian((σ, σ), (blur_window_size, blur_window_size))
    gray = imfilter(gray, kernel)

    #binarize
    blackwhite = binarize(gray, AdaptiveThreshold(window_size=threshold_window_size, percentage=threshold_percentage))
    blackwhite = invert_image(blackwhite)

    blackwhite
end


function construct_grid(height::Int, width::Int; nblocks::Int=3)
    grid = []
    step_i = height/nblocks
    step_j = width/nblocks
    for i in 0:nblocks
        push!(grid, [(step_i * i, 1), (step_i * i, width)])
    end
    for j in 0:nblocks
        push!(grid, [(1, step_j * j), (height, step_j * j)])
    end
    grid
end