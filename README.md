# JuliaCon23_ImageProcessingWorkshop

## Preparation for the workshop (10min)

- If needed, install Julia. We recommend Juliaup:
  - Documentation on installing Juliaup: https://github.com/JuliaLang/juliaup
- If needed, install Jupyter. There are several options:
  - within Julia via [IJulia](https://github.com/JuliaLang/IJulia.jl)
  - using `pip` as described at [Jupyter lab](https://jupyter.org/install)
  - the [Anaconda](https://www.anaconda.com/download/) suite (warning: large)
- Clone this repository with git
- Activate the environment, instantiate, and precompile:
```julia
using Pkg
Pkg.activate(path_to_repository_clone)
Pkg.instantiate()
Pkg.precompile()
```
- Launch Jupyter
- Run the notebooks to ensure they execute (we'll go over them in detail during the workshop)

# JuliaImages resources and social media:

- Documentation: https://juliaimages.org/stable/
- Github: https://github.com/JuliaImages
- Twitter: https://twitter.com/juliaimages
- Mastodon: https://julialang.social/@juliaimages
- Linkedin: https://www.linkedin.com/company/juliaimages
- Youtube: https://www.youtube.com/@JuliaImages
