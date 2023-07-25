
"""
    on_segment(point, segment)

Determine if a point q lies on the segement pr. 
"""
function on_segment(q, segment)
    p, r = segment
    return (
        (q[1] <= max(p[1], r[1])) &&
        (q[1] >= min(p[1], r[1])) &&
        (q[2] <= max(p[2], r[2])) &&
        (q[2] >= min(p[2], r[2]))
        ) && (get_orientation(p, q, r) == 0)
end


"""
    get_orientation(p, q, r)

Determine orientation of three points
- 0 -> co-linear
- 1 -> clockwise
- 2 -> counter-clockwise
"""
function get_orientation(p, q, r)
    cross_product = (q[2] - p[2]) * (r[1] - q[1]) - (r[2] - q[2]) * (q[1] - p[1])
    
    orientation = -1
    if cross_product == 0
        orientation = 0 # co-linear
    elseif cross_product > 0 
        orientation = 1 # clockwise
    else
        orientation = 2 # counter-clockwise
    end
    orientation
end


function do_intersect(segment1, segment2)
    o1 = get_orientation(segment1[1], segment1[2], segment2[1])
    o2 = get_orientation(segment1[1], segment1[2], segment2[2])
    o3 = get_orientation(segment2[1], segment2[2], segment1[1])
    o4 = get_orientation(segment2[1], segment2[2], segment1[2])

    # general case
    if (o1 != o2) && (o3 != o4)
        return true
    end

    # special cases -> co-linear points on the segment
    if ((o1 == 0 && on_segment(segment2[1], segment1)) ||
        (o2 == 0 && on_segment(segment2[2], segment1)) ||
        (o3 == 0 && on_segment(segment1[1], segment2)) || 
        (o4 == 0 && on_segment(segment1[2], segment2))
        )
        return true
    end

    return false
end
    


"""
    point_in_polygon(point, vertices)

Based on "A Simple and Correct Even-Odd Algorithm for the Point-in-Polygon Problem for Complex Polygons" 
by Michael Galetzka and Patrick Glauner (2017). This algorithm is an an extension of the odd-even ray algorithm.
It skips vertices that are on the ray. To compensate, the ray is projected backwards (to the left) so that an 
intersection can be found for a skipped vertix if needed.
"""
function point_in_polygon(point, vertices::AbstractArray, on_border_is_inside=true)
    n =  length(vertices)
    num_intersections = 0

    x = [p[1] for p in vertices]
    extreme_left =  (minimum(x) - 100, point[2])
    extreme_right = (maximum(x) + 100, point[2])

    # step 1: point intersects a vertex or edge
    for i in 1:n
        next_i = (i % n) + 1
        if (point == vertices[i]) || on_segment(point, [vertices[i], vertices[next_i]])
            return on_border_is_inside
        end
    end

    # step 3: check intersections with vertices
    s = 1
    while s <= n
        # step 3a: find a pair of vertices not on the ray
        while (s <= n) && (vertices[s][2] == point[2])
            s += 1
        end
        if s > n
            break # step 2
        end
        next_s = s
        skipped_right = false
        for i in 0:n
            next_s = (next_s) % n + 1
            if vertices[next_s][2] != point[2]
                break
            end
            skipped_right = skipped_right || (vertices[next_s][1] > point[1])
        end
        # step 3b: edge intersect with the ray
        edge = [vertices[s], vertices[next_s]]
        intersect = 0
        if (next_s - s) == 1 || (s==n && next_s ==1) #3b.i
            intersect = do_intersect(edge, [point, extreme_right])
        elseif skipped_right #3b.ii
            intersect = do_intersect(edge, [extreme_left, extreme_right])
        end
        num_intersections += intersect
        s = next_s > s ? next_s : (n + 1)
    end
    return (num_intersections % 2) == 1
end
