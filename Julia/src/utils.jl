using Plots
using Serialization

function clc()
    if Sys.iswindows()
        read(run(`powershell cls`), String)
    elseif Sys.isunix()
        read(run(`clear`), String)
    elseif Sys.islinux()
        read(run(`printf "\033c"`), String)
    end
    nothing
end


function save_plot(plot::Plots.Plot, filename::String)
    try
        open(filename, "w") do io
            serialize(io, plot)
        end
        println("Plot object serialized successfully to $filename.")
    catch e
        println("An error occurred during serialization: $e")
    end
end

function load_plot(filename::String)::Plots.Plot
    try
        plot = open(filename, "r") do io
            deserialize(io)
        end
        println("Plot object deserialized successfully from $filename.")
        return plot
    catch e
        println("An error occurred during deserialization: $e")
        return Plots.plot()  # Return an empty plot object in case of error
    end
end

function save_named_tuple(nt::NamedTuple, filename::String)
    try
        open(filename, "w") do io
            serialize(io, nt)
        end
        println("Named tuple serialized successfully to $filename.")
    catch e
        println("An error occurred during serialization: $e")
    end
end

function load_named_tuple(filename::String)::NamedTuple
    try
        nt = open(filename, "r") do io
            deserialize(io)
        end
        println("Named tuple deserialized successfully from $filename.")
        return nt
    catch e
        println("An error occurred during deserialization: $e")
        return NamedTuple()  # Return an empty named tuple in case of error
    end
end

function applyPlotStyle(; plt, ylabel, xlabel = "x(m)")

    gridstyle = (; grid = true, gridalpha = 0.7, gridlinewidth = 0.4)
    default(;
        linewidth = 2.0,
        top_margin = 2Plots.mm,
        bottom_margin = 5Plots.mm,
        right_margin = 5Plots.mm,
        left_margin = 5Plots.mm,
        gridstyle...,
        # framestyle = :box,
        legendfontsize = 12,
        tickfontsize = 14,
        guidefontsize = 16,
        yminorgrid = false,
        xminorgrid = false,
    )
    pl = plot!(plt; xlabel, ylabel, framestyle = :box)
    display(pl)
    return pl
end

function applyPlotStyle2(; plt)

    gridstyle = (; grid = true, gridalpha = 0.7, gridlinewidth = 0.4)
    default(;
        linewidth = 2.0,
        top_margin = 2Plots.mm,
        bottom_margin = 5Plots.mm,
        right_margin = 5Plots.mm,
        left_margin = 5Plots.mm,
        gridstyle...,
        # framestyle = :box,
        legendfontsize = 12,
        tickfontsize = 14,
        guidefontsize = 16,
        yminorgrid = false,
        xminorgrid = false,
    )
    pl = plot!(plt; framestyle = :box)
    display(pl)
    return pl
end

function change_plot_label(pl; label_num, new_label)
    pl[1][label_num][:label] = new_label
    pl
end

function change_plot_attr(pl; plot_num = 1, attr, new_value)
    pl[1][plot_num][attr] = new_value
    pl
end

function displayAndPushToOverleaf(; pl, name, pushToOverleaf = false)
    display(pl)
    pushToOverleaf && plot_to_overleaf(pl, plotdir(), name * ".pdf")
end

function MyName()
    funcname = string(StackTraces.stacktrace()[3].func)
    contains(funcname, "#") ? split(funcname, "#")[2] : funcname
end

# it returns the name of the variable
macro Name(arg)
    string(arg)
end

function figure_name()
    funcname = MyName()
    figname = contains(funcname, "plot") ? split(funcname, "plot")[2] : funcname
    
end

function read_hammer(filename)
    lines = open(readlines, filename)
    lines = lines[2:end]
    out = (; t=zeros(0), qty=zeros(0))
    

    for line in lines
        tokens = split(line, ",")
        t = parse(Float64, tokens[1])
        qty = parse(Float64, tokens[2]) 
        
        push!(out.t, t)
        push!(out.qty, qty)
    end
    #self.max_liquid_e = max(self.liq_ρ_e, key=lambda tup: tup[0])[0]	 
    out
end

function getTimestamp()
    return string(round(Int, datetime2unix(now())))
end

function ConvertPath(st)
    replace(st, "\\" => "/")
end

# This is min-max scaler for the array between 0.0 and 1.0
# input : x = [1.0, 2.0, 3.0, 4.0, 5.0]
# output : [0.0,  0.25,  0.5,  0.75,  1.0]
function normalizeArray(x)
    upper_limit = maximum(x)
    lower_limit = minimum(x)
    min_max_diff = upper_limit - lower_limit
    shifted_array = @. x - lower_limit
    scaled_array = @. shifted_array / min_max_diff
end

plotdir() = raw"C:\Pardeep\trunk\CO2Transport\shell\HEMWithRealGasEOS\images"
plot_objects_dir() = raw"C:/Pardeep/trunk/CO2Transport/shell/flow-CO2/Julia/output/plot_objects"

"""
    plot_to_overleaf(plt, plotdir, name)

Save figure and push to overleaf.
"""

function plot_to_overleaf(plt, plotdir, name; pull = false, push = true)
    path = joinpath(plotdir, name)
    # println(isnothing(plt))
    savefig(plt, path)
    pull && run(Cmd(`powershell git pull`; dir = plotdir))
    run(Cmd(`powershell git add $name`; dir = plotdir))
    run(Cmd(`powershell git commit -m \"Add/updated plot\"`; dir = plotdir))
    push && run(Cmd(`powershell git push`; dir = plotdir))
end

git_status(plotdir) = run(Cmd(`powershell git status`; dir = plotdir))
git_add(; filename, plotdir) = run(Cmd(`powershell git add $filename`; dir = plotdir))
git_commit(; message, plotdir) = run(Cmd(`powershell git commit -m \"$message\"`; dir = plotdir))
git_pull(plotdir) = run(Cmd(`powershell git pull`; dir = plotdir))
git_push(plotdir) = run(Cmd(`powershell git push`; dir = plotdir))
