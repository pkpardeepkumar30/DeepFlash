module Scalers


function sigmoid(x)
    return 1.0 ./ (1 .+ exp.(-x))
end

# Define the inverse sigmoid (logit) function
function logit(y)
    return log.(y ./ (1.0 .- y))
end

function NoScale(Scale=nothing)
    Scaler(x) = x 
    DeScaler(x) = x
    return Scaler, DeScaler
end

function SimpleScale(Scale)
    
    function Scaler(x) 
        @show x
        @show Scale
        x ./ Scale
    end
    DeScaler(x) = x .* Scale
    return Scaler, DeScaler
end

function SigmoidScale(Scale)
    function Scaler(x)
        x_scaled = x ./ Scale
        sigmoid(x_scaled)
    end

    function DeScaler(x)
        logit(x) .* Scale
    end
    return Scaler, DeScaler
end

function PositivityScale(Scale; start, stop)
    
    function Scaler(x)
        x = x ./ Scale
        # x[start:stop] = 2.0 .* sqrt.(x[start:stop]) 
        x[start:stop] = sqrt.(x[start:stop]) 
        return x
    end

    function DeScaler(x)
        # x[start:stop] = x[start:stop] .^ 2 ./ 4.0        
        x[start:stop] = x[start:stop] .^ 2
        # return x
        return x .* Scale
    end
    return Scaler, DeScaler
end

function LogScale(Scale; start, stop)
    function Scaler(x)
        x_scaled = x ./ Scale
        x_scaled[start:stop] .= log.(x_scaled[start:stop])
        return x_scaled
    end

    function DeScaler(x)
        x[start:stop] .= exp.(x[start:stop])
        x .* Scale
    end
    return Scaler, DeScaler
end

function ZScoreScale(Scale)
    function Scaler(x)
        μ = mean(x)  # Compute mean
        σ = std(x)   # Compute standard deviation
        scaled_x = (x .- μ) ./ σ
        return scaled_x, μ, σ  # Return scaled values, mean, and std deviation
    end

    function DeScaler(x, μ, σ)
        return (x .* σ) .+ μ
    end
end



end