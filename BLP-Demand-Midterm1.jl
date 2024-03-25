cd("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1")
using CSV              
using DataFrames       
using LinearAlgebra    
using Statistics       
using Distributions
using Optim            
using BenchmarkTools   
using Flux
using Random
Random.seed!(98426)
#
# Our data and draws  ------------------------------------------------------------------
num_markets = 200
#market_data = Matrix(undef, num_markets,50)
#for i in 1:size(market_data, 1)
#   market_data[i, :] .= randn(50)
#end

#market_data1 = Matrix(undef, num_markets,500)
#for i in 1:size(market_data, 1)
#   market_data1[i, :] .= randn(500)
#end
#df1 = DataFrame(market_data1, :auto)

#market_data2 = Matrix(undef, num_markets,5000)
#for i in 1:size(market_data, 1)
#   market_data2[i, :] .= randn(5000)
#end
#df2 = DataFrame(market_data2, :auto)

#CSV.write("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws.csv", df, writeheader=false)
#CSV.write("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_500.csv", df1, writeheader=false)
#CSV.write("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_5000.csv", df2, writeheader=false)

v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
#v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_500.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
#v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_5000.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals

v_50 = reshape(v_50, (200,50,1))
#v_50 = reshape(v_50, (200,500,1))
#v_50 = reshape(v_50, (200,5000,1))
blp_data = CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/MIDTERM1_FinalDataSet.csv", DataFrame)
share = Vector(blp_data[!,"share"])
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"marketid"])
firmid = Vector(blp_data[!,"firmid"])

X = Matrix(blp_data[!, ["price","caffeine_score"]])
θ₂ = [0.0]


# Our functions  ------------------------------------------------------------------
function BLP_instruments(X, id, cdid, firmid)
   n_products = size(id,1)
   IV_others = zeros(n_products,1)
   IV_rivals = zeros(n_products,1)
   for j in 1:n_products
       other_index = (firmid.==firmid[j]) .* (cdid.==cdid[j]) .* (id.!=id[j])
       other_x_values = X[other_index,:]
       IV_others[j,:] = sum(other_x_values, dims=1)
       rival_index = (firmid.!=firmid[j]) .* (cdid.==cdid[j])
       rival_x_values = X[rival_index,:]
       IV_rivals[j,:] = sum(rival_x_values, dims=1)
   end
   IV = [X IV_others IV_rivals]
  
   return IV
end

function σ(δ, θ₂, X, v, market_id)
   n_individuals = size(v,2)
   n_products = size(X,1)
   δ = repeat(δ,1,n_individuals)


   μ = zeros(n_products, n_individuals)
   for market in unique(market_id)
       μ[market_id.==market,:] = X[market_id.==market,Not(2)] *(v[market,:,:] .* θ₂)'
   end


   ∑ₖexp = zeros(size(μ))
   for market in unique(market_id)
      denom_sequence = exp.(δ[market_id.==market,:] + μ[market_id.==market,:])
      market_denominator = sum(denom_sequence, dims=1)
      ∑ₖexp[market_id.==market,:] = repeat(market_denominator, sum(market_id.==market))
   end
   𝒯 = exp.(δ+μ) ./ (1 .+ ∑ₖexp)
    σ = mean(𝒯, dims=2)[:]  
   return σ, 𝒯
end

function demand_objective_function(θ₂,X,s,Z,v,market_id)
   δ = zeros(size(s))
   Φ(δ) = δ + log.(s) - log.(σ(δ,θ₂,X,v,market_id)[1])

   tolerance = 1e-6                    
   largest_dif = Inf                    
   max_iterations = 1000                
   counter = 0                          
   while (largest_dif > tolerance)
       δ = Φ(δ)
       largest_dif = maximum(abs.( δ - Φ(δ) ))
       counter += 1
       if counter == max_iterations
           break
       end
   end
   𝒯 = σ(δ,θ₂,X,v,market_id)[2]
   θ₁ = inv((X'Z)*pinv(Z'Z)*(X'Z)') * (X'Z)*pinv(Z'Z)*Z'δ
   ξ = δ - X*θ₁
   W = pinv(Z'Z)
   Q = (Z'ξ)' * W * (Z'ξ)
   return Q, θ₁, ξ, 𝒯
end

function f(θ₂)
   Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
   return Q
end

function ∇(storage, θ₂)
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    g = gradients(θ₂,X,Z,v_50,cdid,ξ,𝒯)
end


function gradients(θ₂,X,Z,v,market_id,ξ,𝒯)
    n_products = size(X,1)
    n_individuals = size(v,2)
    n_coefficients = size(θ₂,1)
    
    W = pinv(Z'Z)
    ∂Q_∂ξ = 2*(Z'ξ)'W*Z'
    
        ∂σᵢ_∂δ = zeros(n_products, n_products, n_individuals)
        diagonal_index = CartesianIndex.(1:n_products, 1:n_products) 
        for individual in 1:n_individuals
            ∂σᵢ_∂δ[:,:,individual] = -𝒯[:,individual] * 𝒯[:,individual]'
            ∂σᵢ_∂δ[diagonal_index, individual] = 𝒯[:,individual] .* (1 .- 𝒯[:,individual])
    
        end
        ∂σ_∂δ = mean(∂σᵢ_∂δ, dims=3)[:,:]
        ∂σ_∂δ⁻¹ = zeros(size(∂σ_∂δ))
        for market in unique(market_id)
            ∂σ_∂δ⁻¹[market_id.==market, market_id.==market] = inv(∂σ_∂δ[market_id.==market, market_id.==market])
        end
        ∂σᵢ_∂θ₂ = zeros(n_products, n_individuals, n_coefficients)
    
        for market in unique(market_id)
            for coef in 1:n_coefficients
                Σⱼx₁ⱼ𝒯ⱼᵢ = X[market_id.==market, coef]' * 𝒯[market_id.==market,:]
                ∂σᵢ_∂θ₂[market_id.==market,:,coef] = v[market,:,coef]' .* 𝒯[market_id.==market,:] .* (X[market_id.==market,coef] .- Σⱼx₁ⱼ𝒯ⱼᵢ)
            end
        end
        ∂σ_∂θ₂ = mean(∂σᵢ_∂θ₂, dims=2)[:,1,:] 
    ∂Q_∂θ₂ = ∂Q_∂ξ * (-∂σ_∂δ⁻¹ * ∂σ_∂θ₂)
    
    return ∂Q_∂θ₂'
end 
    
# Optimization  ------------------------------------------------------------------
Z = Matrix(blp_data[!,["zd1","zd2","zd3","zd4","zd5","zd6","zd7"]])
#X₁ = Matrix(blp_data[!, ["price","caffeine_score"]])
#Z = BLP_instruments(X₁[:,Not(1)], id, cdid, firmid)


# Random search, very slow
result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-3, iterations=500, show_trace=true, show_every=10))
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

# Using the gradient. You are locally convex.
result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(g_tol=1e-6, iterations=100, show_trace=true, show_every=10))  
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

result = optimize(f, ∇, θ₂, BFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

result = optimize(f, ∇, θ₂, GradientDescent(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]

result = optimize(f, ∇, θ₂, ConjugateGradient(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]




