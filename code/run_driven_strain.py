#!/usr/bin/env python3
"""Squeeze perturbed generators and rebuild the ordinary Voronoi diagram."""
import numpy as np
from window_controls import build, select, measure, summarize, save_csv, save_json, fit_rows


def main():
    amplitudes = [0., .01, .02, .05, .10, .20]
    records, scaling = [], []
    for eps in amplitudes:
        for realization in range(40):
            tissue = build(30, .05, 3000+realization, strain=eps)
            selected = select(tissue, 'label_box', 9)
            assert len(selected)==324
            records.append(dict(eps=eps, realization=realization, seed=3000+realization,
                                **measure(tissue, tissue['W'], selected)))
    for eps in [0., .05, .20]:
        for realization in range(25):
            tissue = build(30, .05, 4000+realization, strain=eps)
            for parameter in [4, 6, 8, 10, 12]:
                selected = select(tissue, 'label_box', parameter)
                scaling.append(dict(eps=eps, parameter=parameter, realization=realization,
                                    seed=4000+realization, **measure(tissue, tissue['W'], selected)))
    summary = [summarize([r for r in records if r['eps']==eps], 'eps', eps) for eps in amplitudes]
    scale_summary, fits = [], []
    for eps in [0., .05, .20]:
        group = []
        for parameter in [4, 6, 8, 10, 12]:
            raw = [r for r in scaling if r['eps']==eps and r['parameter']==parameter]
            row = dict(eps=eps, parameter=parameter, N=raw[0]['N'],
                       rms=float(np.sqrt(np.mean([r['regional']**2 for r in raw]))))
            group.append(row); scale_summary.append(row)
        fits.append(fit_rows(group, 'eps', eps))
    save_json('driven.json', summary)
    save_csv('driven_realizations.csv', records)
    save_csv('driven_scaling_realizations.csv', scaling)
    save_csv('driven_scaling_summary.csv', scale_summary)
    save_json('driven_scaling_fits.json', fits)
    print('Rebuilt Voronoi control: 324 cells; 40 paired realizations per amplitude.')
    for r in summary:
        print(f"eps={r['eps']:.2f}: gap={r['regional']:.8f}; C_amp={r['corr']:.6f}; centered={r['corr_centered']:.6f}")


if __name__ == '__main__':
    main()
