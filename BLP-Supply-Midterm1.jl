cd("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1")
using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean
using Distributions
using Optim             # for minimization functions
using BenchmarkTools    # for timing/benchmarking functions
using HypothesisTests

# Our data and draws  ------------------------------------------------------------------
v_5000 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_5000.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
v_5000 = reshape(v_5000, (200,5000,1))

v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
#v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_500.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
#v_50 = Matrix(CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/draws_5000.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals

v_50 = reshape(v_50, (200,50,1))
#v_50 = reshape(v_50, (200,500,1))
#v_50 = reshape(v_50, (200,5000,1))

blp_data = CSV.read("/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/MIDTERM1_FinalDataSet.csv", DataFrame)
share = Vector(blp_data[!,"share"])
id = Vector(blp_data[!,"id"])
firm_id = Vector(blp_data[!,"firmid"])
cdid = Vector(blp_data[!,"marketid"])
market_id = Vector(blp_data[!,"marketid"])


X = Matrix(blp_data[!, ["costattributes"]])
x₁= Matrix(blp_data[!, ["price", "caffeine_score"]])
P = Vector(blp_data[!, "price"])
S = Vector(blp_data[!, "share"])
θ₂ = [-0.133]
θ₁ = [-1.62,2.12]


# Our functions  ------------------------------------------------------------------
function price_elasticities(θ₁, θ₂, X, s, v_diag, v_off_diag, market_id, firm_id)
   α = θ₁[1]
   σᵛₚ = θ₂[1]
  
   n_products = size(X,1)
  
   Δ = zeros(n_products, n_products)
  
   #=
   Own price elasticity:
   ∂σⱼ/∂pⱼ = ∫(α + vᵢₖₚσᵛₚ) 𝒯ⱼ (1 - 𝒯ⱼ) f(vᵢ)vᵢ
   Corresponds to the diagonal of Δ
   =#
  
   # X is a vector of observables for all products for all markets
   # vᵢ is a vector of 5 random draws for a given individual
   # j is a particular product
   # recall there is no random coefficient for space (index 6 of X)
   # note that there are about ~100 products per market
  
   # loop through all products
   Threads.@threads for j in 1:n_products # run loop in parallel with Threads. reduced time ~75x.
       market = market_id[j]
  
       # get observables and indiviudals
       xⱼ = X[j,:]                    # observables for product j
       xₘ = X[market_id.==market,:]   # observables of all products in market with product j
       vₘ = v_diag[market,:,:]        # matrix of 5000x5 pre-selected random draws (=> 5000 individuals)
     
       # function defining the interior of the sigma function integral
       F(vᵢ) = exp(xⱼ'θ₁ + xⱼ[Not(2)]'*(θ₂.*vᵢ)) / (1 + sum(exp.(xₘ*θ₁ + xₘ[:,Not(2)]*(θ₂.*vᵢ))))
  
       # interior of the own price elasticity function
       integral_interior(vᵢ) = (α + vᵢ[1]*σᵛₚ) * F(vᵢ) * (1 - F(vᵢ))
  
       # estimate with Monty Carlo integration over all individuals in vₘ
       # integral_interior() is applied to each of the ~5000 sets of 5 vᵢ values in vₘ
       ∂σⱼ_∂pⱼ = mean(integral_interior.(vₘ))
       # equivalently: ∂σⱼ_∂pⱼ = sum(integral_interior.(vₘ)) * 1 / length(vₘ)
  
       # assign own price elasticitiy to matrix of price elasticities (along the diagonal)
       Δ[j,j] = -∂σⱼ_∂pⱼ
  
   end
  
  
   #=
   Cross price elasticity:
   ∂σⱼ/∂pₖ ∫ - (α + vᵢₖₚσᵛₚ) 𝒯ⱼ 𝒯ₖ f(vᵢ)vᵢ
   =#
  
   # X is a vector of observables for all products for all markets
   # vᵢ is a vector of 5 random draws for a given individual
   # j and k are particular products
   # recall there is no random coefficient for space (index 6 of X)
   # note that there are about ~100 products per market
  
   # loop through all columns (σ)
   Threads.@threads for j in 1:n_products  # run loop in parallel with Threads. reduced time ~500x.
       # loop through all rows (price)
       for k in 1:n_products
  
           # check that the row and column product are both marketed by the same company in the same market
           if (firm_id[j] == firm_id[k]) & (market_id[j] == market_id[k]) & (j != k)
  
               xⱼ = X[j,:]                        
               xₖ = X[k,:]                        
               xₘ = X[market_id.==market_id[j],:] 
               vₘ = v_off_diag[market_id[j],:,:]    
  
               F(xⱼ,vᵢ) = exp(xⱼ'θ₁ + xⱼ[Not(2)]'*(θ₂.*vᵢ)) / (1 + sum(exp.(xₘ*θ₁ + xₘ[:,Not(2)]*(θ₂.*vᵢ))))
  
               integral_interior(vᵢ) = (α + vᵢ[1]*σᵛₚ) * F(xⱼ,vᵢ) * F(xₖ,vᵢ)
  
               ∂σⱼ_∂pₖ = mean(integral_interior.(vₘ))
              
               Δ[k,j] = -∂σⱼ_∂pₖ
           end
       end
   end
   return Δ
end


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

# Instruments ------------------------------------------------------------------
X₁ = Matrix(blp_data[!, ["price","caffeine_score"]])
Z = BLP_instruments(X₁[:,Not(1)], id, cdid, firmid)
#Z = Matrix(blp_data[!,["zs1","zs2","zs3","zs4","zs5","zs6","zs7"]])

# Finding θ₃ with 2SLS ------------------------------------------------------------------
Δ = price_elasticities(θ₁, θ₂, x₁, S, v_5000, v_50, market_id, firm_id)
Δ⁻¹ = inv(Δ)
MC = P - Δ⁻¹*S
neg_MC = 0
for i in 1:size(MC,1)
    if MC[i] < 0
       neg_MC = neg_MC +1
    end
end
has_negative = any(MC .< 0)
neg_MC
IV = [Z X]
𝓧 = IV*pinv(IV'IV)*IV'X
𝓧 = 𝓧[MC.>0,:]
MC = MC[MC.>0]

θ₃ = inv(𝓧'𝓧)*𝓧'*log.(MC)


# Solving for baseline price  ------------------------------------------------------------------
MC = P - Δ⁻¹*S
ω = MC- X*θ₃
blp_data[!, :ω] = vec(ω)
MCB = X*θ₃+ω 
X = Matrix(blp_data[!, ["price","caffeine_score"]])
P = Matrix(blp_data[!, ["price"]])
function sharefunction(θ₁ ,θ₂, X, v, market_id,P)
    n_individuals = size(v,2)
    n_products = size(X,1)
    δ = X*θ₁
    μ = zeros(n_products, n_individuals)
    δ = repeat(δ,1,n_individuals)
    for market in unique(market_id)
        μ[market_id.==market,:] = P[market_id.==market] *(v[market,:,:] .* θ₂)'
    end
    ∑ₖexp = zeros(size(μ))
    for market in unique(market_id)
       denom_sequence = exp.(δ[market_id.==market,:] + μ[market_id.==market,:])
       market_denominator = sum(denom_sequence, dims=1)
       ∑ₖexp[market_id.==market,:] = repeat(market_denominator, sum(market_id.==market))
    end
    𝒯 = exp.(δ+μ) ./ (1 .+ ∑ₖexp)
    σ = mean(𝒯, dims=2)[:]  
    return σ
end
function objective(P)
    return norm(sharefunction(θ₁ ,θ₂, X, v_50, market_id,P) - Δ * (P - MCB))
end
P0 = Matrix(blp_data[!, ["price"]])
result = optimize(objective, P0, NelderMead())
BaselineP = Optim.minimizer(result)
BaselineShare = sharefunction(θ₁ ,θ₂, X, v_50, market_id,BaselineP)

# Comparing Baseline price and baseline share to observed price and share
blp_data[!, :P] = vec(BaselineP)
blp_data[!, :s] = vec(BaselineShare)
differences = vec(BaselineP - P)
result = OneSampleTTest(differences)
pvalue(result)
#=
pvalue(result)
8.048640529753332e-16> 0.05
=#

differences = vec(BaselineShare - S)
result = OneSampleTTest(differences)
pvalue(result)
#=
julia> pvalue(result)
0.12910673022676675> 0.05
=#

df_filtered = filter(row -> row.firmid == 1 || row.firmid == 2, blp_data)
BaselineData = df_filtered[!, ["firmid","P","s","marketid","id"]]


# Merger Function Efficient  ------------------------------------------------------------------
#filter the for id == 1 or 2
df_filtered = filter(row -> row.firmid == 1 || row.firmid == 2, blp_data)
# Grouping the filtered DataFrame by the column marketid
df_grouped = groupby(df_filtered, :marketid)
# Calculating the minimum value of ω within each group
df_min_ω = combine(df_grouped, :ω => minimum => :ω_AM)
# Add it to the original data
blp_data = leftjoin(blp_data, df_min_ω, on = :marketid)
# Modify it for firmid == 3
transform!(blp_data, [:ω, :ω_AM, :firmid] => ByRow((ω, ω_AM, firmid) -> firmid == 3 ? ω : ω_AM) => :ω_AM)

X = Matrix(blp_data[!, ["caffeine_scoreAM"]])
ω = Matrix(blp_data[!, ["ω_AM"]])
MC = X*θ₃+ω 
# Create a single id for the new firm
transform!(blp_data, :firmid => ByRow(firmid -> firmid == 2 ? 1 : firmid) => :firmid)

firm_id = Vector(blp_data[!,"firmid"])
market_id = Vector(blp_data[!,"marketid"])
X = Matrix(blp_data[!, ["price","caffeine_score"]])
Δ = price_elasticities(θ₁, θ₂, x₁, S, v_5000, v_50, market_id, firm_id)
function objective(P)
    return norm(sharefunction(θ₁ ,θ₂, X, v_50, market_id,P) - Δ * (P - MC))
end
result1 = optimize(objective, P0, NelderMead())
P1 = Optim.minimizer(result1)
Share1 = sharefunction(θ₁ ,θ₂, X, v_50, market_id,P1)
blp_data[!, :P1] = vec(P1)
blp_data[!, :s1] = vec(Share1)
df_filtered = filter(row -> row.firmid == 1, blp_data)
df_grouped = groupby(df_filtered, :marketid)
df_sum = combine(df_grouped, :s1 => sum => :s1_sum)
EfficientData = df_filtered[!, ["firmid","P1","s1","marketid","id"]]

#Testing the Efficient Merger for id =1 so MD
df_filtered_id1 = filter(row -> row.id == 1, EfficientData)
P1_id1 = df_filtered_id1.P1
df_filtered_id1 = filter(row -> row.id == 1, BaselineData)
BaselineP_id1 = df_filtered_id1.P
differences = vec(P1_id1 - BaselineP_id1)
result = OneSampleTTest(differences)
pvalue(result, tail=:right)

#Testing the Efficient Merger for id =2 so MD
df_filtered_id2 = filter(row -> row.id == 2, EfficientData)
P1_id2 = df_filtered_id2.P1
df_filtered_id2 = filter(row -> row.id == 2, BaselineData)
BaselineP_id2 = df_filtered_id2.P
differences = vec(P1_id2 - BaselineP_id2)
result = OneSampleTTest(differences)
pvalue(result, tail=:right)


# Merger Function Average ------------------------------------------------------------------
df_filtered = filter(row -> row.firmid == 1 || row.firmid == 2, blp_data)
df_grouped = groupby(df_filtered, :marketid)
df_avg_ω = combine(df_grouped, :ω => mean => :ω_AV)  # Compute the average of ω within each group
blp_data = leftjoin(blp_data, df_avg_ω, on = :marketid)
transform!(blp_data, [:ω, :ω_AV, :firmid] => ByRow((ω, ω_AV, firmid) -> firmid == 3 ? ω : ω_AV) => :ω_AV)

X = Matrix(blp_data[!, ["caffeine_scoreAM2"]])
ω = Matrix(blp_data[!, ["ω_AV"]])
MC = X*θ₃+ω 
firm_id = Vector(blp_data[!,"firmid"])
market_id = Vector(blp_data[!,"marketid"])
X = Matrix(blp_data[!, ["price","caffeine_score"]])
Δ = price_elasticities(θ₁, θ₂, x₁, S, v_5000, v_50, market_id, firm_id)
function objective(P)
    return norm(sharefunction(θ₁ ,θ₂, X, v_50, market_id,P) - Δ * (P - MC))
end
result2 = optimize(objective, P0, NelderMead())
P2= Optim.minimizer(result2)
Share2 = sharefunction(θ₁ ,θ₂, X, v_50, market_id,P2)
blp_data[!, :P2] = vec(P2)
blp_data[!, :s2] = vec(Share2)
df_filtered = filter(row -> row.firmid == 1, blp_data)
AverageData = df_filtered[!, ["firmid","P2","s2","marketid","id"]]

#Testing the Average Merger for id =1 so MD
df_filtered_id1 = filter(row -> row.id == 1, AverageData)
P2_id1 = df_filtered_id1.P2
df_filtered_id1 = filter(row -> row.id == 1, BaselineData)
BaselineP_id1 = df_filtered_id1.P
differences = vec(P2_id1 - BaselineP_id1)
result = OneSampleTTest(differences)
pvalue(result, tail=:right)

#Testing the Average Merger for id =2 so MD
df_filtered_id2 = filter(row -> row.id == 2, AverageData)
P2_id2 = df_filtered_id2.P2
df_filtered_id2 = filter(row -> row.id == 2, BaselineData)
BaselineP_id2 = df_filtered_id2.P
differences = vec(P2_id2 - BaselineP_id2)
result = OneSampleTTest(differences)
pvalue(result, tail=:right)



differences = vec(P1_id1 - P2_id1)
result = OneSampleTTest(differences)
pvalue(result)


differences = vec(P1_id2 - P2_id2)
result = OneSampleTTest(differences)
pvalue(result)
