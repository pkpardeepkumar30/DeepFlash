module CO2Transport
# include("Solvers.jl")

include("MultiComponent.jl")
# include("MultiComponent.jl")
include("utils.jl")



# export TimeIntegrator
# export TestTanks
export clc, figure_name, Name
export save_plot, load_plot, save_named_tuple, load_named_tuple
export save_module_state, load_module_state
export change_plot_label, change_plot_attr, applyPlotStyle, applyPlotStyle2
# export MyName
export normalizeArray
export git_status,  git_push, git_pull, git_add, git_commit, plot_to_overleaf, plotdir
export plot_objects_dir, displayAndPushToOverleaf
export MultiComponent
# export Thermodynamics
end