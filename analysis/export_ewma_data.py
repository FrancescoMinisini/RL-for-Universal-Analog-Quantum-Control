import pandas as pd
import numpy as np
import os

# Paths are relative to the uqc_repro root, whatever the working directory.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Column prefix in the combined CSV -> method subdirectory written by benchmark_robustness.py.
# The Adam plan evaluated in each sweep lives under a different subdirectory name.
#
# WARNING: in robustness_analysis/ and robustness_analysis_2/ the "noise" plan
# (final_results/noise/) targets N(0, 0, pi/2), not N(2.2, 2.2, pi/2), and the "adam" plan of
# robustness_analysis_2/ (final_results/adam_test/) targets N(0.2, 0.2, pi/2); each is scored
# against its own target. Those two sweeps are kept only because the submitted thesis plots
# them. robustness_analysis_3/ evaluates only plans for N(2.2, 2.2, pi/2) and is the one the
# preprint uses.
SWEEPS = {
    "final_results/robustness_analysis": {"adam": "adam_noise", "nominal": "nominal", "noise": "noise"},
    "final_results/robustness_analysis_2": {"adam": "adam_test", "nominal": "nominal", "noise": "noise"},
    "final_results/robustness_analysis_3": {
        "adam70": "adam_70ns",
        "adam60": "adam_60ns",
        "nominal": "trpo_nominal",
        "noise": "trpo_noise",
        "noise91": "trpo_noise_it091",
    },
}


def export_combined_data(base_dir, methods, out_file, span_fid=50, span_var=80):
    dfs = []
    for m, subdir in methods.items():
        path = os.path.join(base_dir, subdir, "robustness_curve.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Compute EWMA on individual arrays
            df[f'{m}_fidelity_raw'] = df['average_fidelity']
            df[f'{m}_fidelity_ewma'] = df['average_fidelity'].ewm(span=span_fid).mean()

            df[f'{m}_variance_raw'] = df['fidelity_variance']
            df[f'{m}_variance_ewma'] = df['fidelity_variance'].ewm(span=span_var).mean()

            df = df[['sigma_mhz', f'{m}_fidelity_raw', f'{m}_fidelity_ewma', f'{m}_variance_raw', f'{m}_variance_ewma']]
            df = df.set_index('sigma_mhz')
            dfs.append(df)

    if not dfs:
        print(f"No data found in {base_dir}")
        return

    combined = pd.concat(dfs, axis=1)

    # Sort just in case
    combined = combined.sort_index()

    combined.to_csv(out_file)
    print(f"Exported combined EWMA data to {out_file}")

if __name__ == "__main__":
    for base_dir, methods in SWEEPS.items():
        export_combined_data(base_dir, methods, os.path.join(base_dir, "combined_ewma_data.csv"))
