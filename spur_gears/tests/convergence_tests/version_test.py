# 1. interpolate never, use Vm
main(mesh_sizes, p_deg, interpolate="never", use_Vm=True, mesh_all_magnets=ma, \
        D_ref=D, dir=f"Vm_interpol_never_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
# 2. interpolate once, use Vm
main(mesh_sizes, p_deg, interpolate="once", use_Vm=True, mesh_all_magnets=ma, \
        D_ref=D, dir=f"Vm_interpol_once_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
# 3. interpolate twice, use Vm

ct = create_convergence_test()
errors, names = convergence_test(ct, mesh_size=mesh_size, p_deg=p_deg, \
                                    interpolate="twice", use_Vm=True, \
                                    mesh_all_magnets=False, D_ref=D, \
                                        analytical_solution=True)

# 4. interpolate never, use B directly
main(mesh_sizes, p_deg, interpolate="never", use_Vm=False, mesh_all_magnets=ma, \
        D_ref=D, dir=f"B_interpol_never_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
# 5. interpolate once, use B directly
main(mesh_sizes, p_deg, interpolate="once", use_Vm=False, mesh_all_magnets=ma, \
        D_ref=D, dir=f"B_interpol_once_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
# 6. interpolate twice, use B directly
main(mesh_sizes[1:], p_deg, interpolate="twice", use_Vm=False, mesh_all_magnets=ma, \
        D_ref=D, dir=f"B_interpol_twice_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)

print(f"In gear 1 {ct.gear_1.n - len(ct.gear_1.magnets)} out of {ct.gear_1.n} magnets have been deleted.")
print(f"In gear 2 {ct.gear_2.n - len(ct.gear_2.magnets)} out of {ct.gear_2.n} magnets have been deleted.")

for error, name in zip(errors, names):
    with open(f"{name}.csv", "a+") as f:
        f.write(f"{mesh_size} {error} {D} {ct.gear_1.R} {ct.gear_2.R} {D - ct.gear_1.R - ct.gear_2.R} \n")