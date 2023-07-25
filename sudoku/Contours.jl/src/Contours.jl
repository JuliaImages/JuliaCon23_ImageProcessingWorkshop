
module Contours

export find_contours,

    point_in_polygon,
    area_polygon, centroid_polygon,
    centroid_contour, area_contour,

    draw_contour!, draw_contours!,
    fill_contour!, fill_contours!,

    AbstractFillAlgorithm, Boundary4Fill, ScanFill


include("find_contours.jl")
include("point_in_polygon.jl")
include("moments.jl")
include("draw.jl")

area_contour(contour::AbstractArray{CartesianIndex}) = area_polygon(contour)
centroid_contour(contour::AbstractArray{CartesianIndex}) = centroid_polygon(contour)

end # module Contours
