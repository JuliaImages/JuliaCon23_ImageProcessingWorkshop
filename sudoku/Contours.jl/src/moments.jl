"""
    area_polygon(vertices)

Uses the shoelace formula: Σ(xᵢyᵢ₊₁ - yᵢxᵢ₊₁)
"""
function area_polygon(vertices::AbstractArray)
    return abs(first_moment(vertices))
end


"""
    centroid_polygon(vertices)

Uses the following formulas:
- Cx = (1/6A)Σ(xᵢ + xᵢ₊₁)(xᵢyᵢ₊₁ - yᵢxᵢ₊₁)
- Cy = (1/6A)Σ(yᵢ + yᵢ₊₁)(xᵢyᵢ₊₁ - yᵢxᵢ₊₁)   
"""
function centroid_polygon(vertices::AbstractArray)
    area = first_moment(vertices)
    n = length(vertices)
    Cx = 0.0
    Cy = 0.0
    for i in 1:n
        ptᵢ = vertices[i]
        ptᵢ₊₁ = vertices[(i % n) + 1]
        a = (ptᵢ[1] * ptᵢ₊₁[2] - ptᵢ[2] * ptᵢ₊₁[1])
        Cx += (ptᵢ[1] + ptᵢ₊₁[1]) * a
        Cy += (ptᵢ[2] + ptᵢ₊₁[2]) * a
    end
    return (Cx/(6 * area), Cy/(6 * area))
end


function first_moment(veritices::AbstractArray)
    # first moment of a simple polygon
    n = length(veritices)
    moment = 0.0
    for i in 1:n
        ptᵢ = veritices[i]
        ptᵢ₊₁ = veritices[(i % n) + 1]
        moment += ptᵢ[1] * ptᵢ₊₁[2] - ptᵢ[2] * ptᵢ₊₁[1]
    end
    moment *= 0.5
    return moment
end
