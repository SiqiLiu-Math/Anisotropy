#!/usr/bin/env python3
"""Empirical finite-range comparison of masks in nonorthogonal lattice labels.

A label disk is NOT a physical disk; rotating a label box changes its physical
shape. These experiments do not isolate a commensurability mechanism.
"""
import numpy as np
from window_controls import ROOT, BASIS, build, select, boundary_stats, save_csv, save_json, fit_rows

SPECS = {'label_box':[3,4,5,6,7,8,9,10],
         'label_disk':[3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5],
         'label_rotated_box':[3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5]}


def main():
    records, masks = [], {}
    for realization in range(60):
        tissue = build(34, .05, 2000+realization)
        for name, parameters in SPECS.items():
            for parameter in parameters:
                selected = select(tissue, name, parameter)
                P, nb = boundary_stats(tissue, selected)
                w = sum(tissue['W'][i] for i in selected)
                records.append(dict(name=name, param=parameter, realization=realization,
                                    seed=2000+realization, N=len(selected), P=P, nb=nb,
                                    sum_W_real=float(w.real), sum_W_imag=float(w.imag), gap=float(abs(w)),
                                    area=float(sum(tissue['area'][i] for i in selected))))
                key = name+'_'+str(parameter).replace('.', '_')
                if realization==0:
                    masks[key] = dict(kind=name, parameter=parameter,
                                      selected_global_indices=selected,
                                      selected_lattice_labels=tissue['ij'][selected].astype(int).tolist())
    rows, fits = [], []
    for name, parameters in SPECS.items():
        group = []
        for parameter in parameters:
            raw = [r for r in records if r['name']==name and r['param']==parameter]
            row = dict(name=name, param=parameter, N=float(np.mean([r['N'] for r in raw])),
                       P=float(np.mean([r['P'] for r in raw])), nb=float(np.mean([r['nb'] for r in raw])),
                       rms=float(np.sqrt(np.mean([r['gap']**2 for r in raw]))))
            rows.append(row); group.append(row)
        f = fit_rows(group, 'name', name)
        f['boundary_cell_slope'] = float(np.polyfit(np.log([r['nb'] for r in group]), np.log([r['rms'] for r in group]), 1)[0])
        fits.append(f)
    save_json('boundary_scaling.json', rows)
    save_json('boundary_scaling_fits.json', fits)
    save_csv('boundary_scaling_realizations.csv', records)
    save_json('window_control_protocol.json', dict(
        basis_rows=BASIS.tolist(), boundary_n=34, boundary_eta=.05, boundary_center=[16.5,16.5],
        boundary_seeds=list(range(2000,2060)), label_rotation_degrees=31., masks=masks,
        loading_n=30, loading_eta=.05, loading_center=[14.5,14.5], loading_halfwidth=9,
        loading_actual_N=324, loading_selection='abs(i-14.5)<=9 and abs(j-14.5)<=9; labels i,j=6,...,23',
        loading_scaling_halfwidths=[4,6,8,10,12], loading_scaling_N=[64,144,256,400,576],
        strain_seeds=list(range(3000,3040)), strain_scaling_seeds=list(range(4000,4025)),
        polarized_seeds=list(range(5000,5040)), polarized_scaling_seeds=list(range(6000,6025)),
        physical_window_caveat='All masks are in nonorthogonal reference lattice labels. Physical shape and orientation vary together.',
        C_amp='sum Re(Wa conj(Wb)) / sum abs(Wa)*abs(Wb), then arithmetic mean over realizations; uncentered',
        C_amp_centered='same ratio after subtracting each observation-window complex mean W from every cell',
        C_centered_rms='centered numerator / sqrt(sum abs(Wa-meanW)^2 * sum abs(Wb-meanW)^2)',
        neighbors='three positive reference-label offsets (1,0),(0,1),(-1,1); not a recomputed graph shell',
        null_ratio='ensemble RMS regional gap / ensemble mean sqrt(sum abs(Wa)^2); random independent orientation comparison, not 1=no cancellation',
        interpretation='Empirical finite-range fits only; not an asymptotic law or mechanistic classifier. Fixed-geometry modulated tensions are not re-equilibrated.'
    ))
    for fit in fits:
        print(f"{fit['name']}: empirical slope vs N={fit['fitted_slope']:.6f}; vs boundary count={fit['boundary_cell_slope']:.6f}")


if __name__ == '__main__':
    main()
