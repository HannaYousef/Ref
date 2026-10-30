# Extract CL, CD and the trim angle of attack of the V3 kite as functions of depower
# from the full V3Kite.jl model (deforming wing, VSM aerodynamics, bridle, KCU, tether).
#
# Run this on a machine with V3Kite.jl installed (>= 16 GB RAM):
#
#   cd V3Kite.jl
#   cp <SoftKiteReference>/scripts/v3kite_extract_coeffs.jl examples/
#   ./bin/run_julia
#   julia> include("examples/v3kite_extract_coeffs.jl")
#
# It parks the kite at constant tether length for each depower setting (the same
# init/step! interface as examples/simple_parking.jl), averages the settled state and
# writes output/v3kite_coeffs.csv. Then, in SoftKiteReference:
#
#   python scripts/fit_v3kite_logs.py <path>/v3kite_coeffs.csv
#
# which writes softkite/data/v3_aero_v3kite.yaml (same format as softkite/data/v3_aero.yaml, with the
# depower axis in u_p instead of the line shortening).
#
# NOTE: written against the V3Kite.jl API of October 2026 (init, step!, span_mean_aoa,
# compute_lift_coeff, compute_drag_coeff, winch_adapter.jl). It could not be run where
# it was written (no Julia there), so check the first case before a long sweep.

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: init, step!
using LinearAlgebra: norm
using Printf: @printf
using Statistics: mean
include(joinpath(@__DIR__, "winch_adapter.jl"))  # winch_pos_controller, winch_torque!

# ==================== USER PARAMETERS ==================== #
PROJECT       = "system_reelout.yaml"
V_WIND        = 9.51                  # wind speed at reference height [m/s]
TETHER_LENGTH = 150.0                 # [m]
DEPOWERS      = 0.20:0.025:0.40       # relative depower settings to identify [-]
SIM_TIME      = 20.0                  # per case [s]
T_AVG         = 8.0                   # averaging window at the end of each case [s]
DT            = 0.05 / 3              # [s]
REL_STEERING  = 0.004                 # steering trim (see simple_parking.jl)
OUT_FILE      = joinpath(@__DIR__, "..", "output", "v3kite_coeffs.csv")

rad2deg_(x) = x * 180 / pi

function run_case(depower)
    s = init(V_WIND, TETHER_LENGTH; depower_setpoint = depower, sim_time = SIM_TIME,
             dt = DT, system_yaml = PROJECT)
    s.sys.winches[1].brake = true
    l0 = s.sys_state.l_tether[1]
    wpc = winch_pos_controller(s)
    rows = NamedTuple[]
    for _ in 1:s.steps
        step!(s; rel_depower = depower, rel_steering = REL_STEERING,
              set_torque = winch_torque!(wpc, s, l0))
        t = s.sys_state.time
        t < SIM_TIME - T_AVG && continue
        sys = s.sam.sys_struct
        push!(rows, (aoa_mean = rad2deg_(span_mean_aoa(sys)),
                     aoa_center = rad2deg_(compute_kite_aoa(sys)),
                     cl = compute_lift_coeff(s.sam),
                     cd = compute_drag_coeff(s.sam),        # non-tether CD (VSM + parasitic)
                     cd_tether = compute_tether_drag_coeff(s.sam),
                     v_app = norm(s.sys_state.v_app),
                     force = s.sys_state.force[1],
                     elevation = rad2deg_(s.sys_state.elevation),
                     heading = rad2deg_(s.sys_state.heading)))
    end
    m(f) = mean(getfield.(rows, f))
    return (depower = depower, aoa_mean = m(:aoa_mean), aoa_center = m(:aoa_center),
            cl = m(:cl), cd = m(:cd), cd_tether = m(:cd_tether), v_app = m(:v_app),
            force = m(:force), elevation = m(:elevation), heading = m(:heading), n = length(rows))
end

mkpath(dirname(OUT_FILE))
open(OUT_FILE, "w") do io
    println(io, "depower,aoa_mean_deg,aoa_center_deg,cl,cd,cd_tether,v_app,force,elevation_deg,heading_deg,n,v_wind,l_tether")
    for dp in DEPOWERS
        r = try
            run_case(dp)
        catch e
            @warn "case depower=$dp failed" exception = e
            continue
        end
        @printf("depower %.3f: AoA %.2f deg  CL %.3f  CD %.3f  v_app %.1f m/s  F %.0f N  el %.1f deg\n",
                r.depower, r.aoa_mean, r.cl, r.cd, r.v_app, r.force, r.elevation)
        println(io, join((r.depower, r.aoa_mean, r.aoa_center, r.cl, r.cd, r.cd_tether, r.v_app,
                          r.force, r.elevation, r.heading, r.n, V_WIND, TETHER_LENGTH), ","))
        flush(io)
    end
end
@info "wrote $OUT_FILE"
