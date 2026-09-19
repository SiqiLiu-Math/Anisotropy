#!/usr/bin/env python3
"""Coefficient modulation on fixed Voronoi cells; NOT re-equilibrated mechanics."""
import numpy as np
from window_controls import (build, polarized_W, select, measure, summarize,
                             save_csv, save_json, fit_rows, junction_residual)


def main():
    amplitudes = [0., .02, .05, .10, .20, .40]
    records, scaling = [], []
    for realization in range(40):
        tissue = build(30, .05, 5000+realization)
        selected = select(tissue, 'label_box', 9)
        assert len(selected) == 324
        for beta in amplitudes:
            W = polarized_W(tissue, beta)
            records.append(dict(beta=beta, realization=realization, seed=5000+realization,
                                max_relative_junction_residual=junction_residual(tissue, beta, selected),
                                **measure(tissue, W, selected)))
    for realization in range(25):
        tissue = build(30, .05, 6000+realization)
        for beta in [0., .10, .40]:
            W = polarized_W(tissue, beta)
            for parameter in [4, 6, 8, 10, 12]:
                selected = select(tissue, 'label_box', parameter)
                scaling.append(dict(beta=beta, parameter=parameter, realization=realization,
                                    seed=6000+realization, **measure(tissue, W, selected)))
    summary = []
    for beta in amplitudes:
        r = summarize([x for x in records if x['beta']==beta], 'beta', beta)
        r['reg'] = r.pop('regional')  # Historical plotting key retained.
        r['regular_hexagon_gap'] = float(324*np.sqrt(12)*beta)
        summary.append(r)
    scale_summary, fits = [], []
    for beta in [0., .10, .40]:
        group = []
        for parameter in [4, 6, 8, 10, 12]:
            raw = [r for r in scaling if r['beta']==beta and r['parameter']==parameter]
            row = dict(beta=beta, parameter=parameter, N=raw[0]['N'],
                       rms=float(np.sqrt(np.mean([r['regional']**2 for r in raw]))))
            group.append(row); scale_summary.append(row)
        fits.append(fit_rows(group, 'beta', beta))
    save_json('polarized.json', summary)
    save_csv('polarized_realizations.csv', records)
    save_csv('polarized_scaling_realizations.csv', scaling)
    save_csv('polarized_scaling_summary.csv', scale_summary)
    save_json('polarized_scaling_fits.json', fits)
    print('Fixed-geometry diagnostic: 324 cells, 40 realizations/amplitude; tensions NOT re-equilibrated.')
    print('Saved original uncentered amplitude-weighted orientation statistic and mean-subtracted controls.')
    for r in summary:
        print(f"beta={r['beta']:.2f}: gap={r['reg']:.8f}; C_amp={r['corr']:.6f}; centered={r['corr_centered']:.6f}")


if __name__ == '__main__':
    main()
