"""Commands recorded for each guide chapter (see record.py).

Each step: name, cmd, optional mode ("host" runs on the Mac outside the container),
show ("tail" keeps only the last n lines), timeout, stop_on_error.
"""
T = "T=datasets/raw/tches20/TCHES20V3_CNN_SCA"

PLANS = {
    "00_start_here": [
        {"name": "docker_build", "cmd": "docker build -t bi-certification .", "mode": "host", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "docker_levelA", "cmd": 'mkdir -p results && docker run --rm -v "$PWD/results:/artifact/results" bi-certification', "mode": "host", "show": "tail", "n": 24},
        {"name": "docker_shell", "cmd": 'mkdir -p results datasets/raw checkpoints external && docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" bi-certification bash -c "pwd && ls"', "mode": "host"},
    ],
    "01_paper_results": [
        {"name": "levelA", "cmd": "python -m paper_results.reproduce archived", "show": "tail", "n": 24},
        {"name": "levelA_checks", "cmd": "python -m paper_results.reproduce archived --skip-oracle 2>/dev/null | grep -E '^\\[(PASS|DIFF)\\]'"},
        {"name": "levelA_files", "cmd": "ls results/paper_results"},
        {"name": "table7", "cmd": "cat results/paper_results/table7_bi_suite.md"},
        {"name": "table8", "cmd": "cat results/paper_results/table8_key_recovery.md"},
        {"name": "table9", "cmd": "cat results/paper_results/table9_biloc_full_suite.md"},
        {"name": "fetch_tches20", "cmd": "paper_results/fetch_tches20.sh", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "levelB", "cmd": "python -m paper_results.reproduce tches20", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "levelB_counts", "cmd": "ls results/paper_results/recomputed/tches20_suite && head -5 results/paper_results/recomputed/tches20_suite/ascad_desync_0_attack_success.csv"},
        {"name": "fetch_estranet", "cmd": "paper_results/fetch_estranet.sh", "show": "tail", "n": 24},
        {"name": "checkpoints", "cmd": "sha256sum bi_certification_checkpoints.tar.gz && tar -xzf bi_certification_checkpoints.tar.gz && ls checkpoints/estranet checkpoints/keras_ascad_random_key"},
        {"name": "fetch_ascad_random", "cmd": "python -m paper_results.fetch_ascad_random_key", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "ches_file", "cmd": "ls -l datasets/raw/ches_ctf_2025 && sha256sum datasets/raw/ches_ctf_2025/CHES_Challenge.h5", "timeout": 3600},
        {"name": "levelC", "cmd": "python -m paper_results.reproduce estranet --suites ascad_random_key ches_ctf_2025", "show": "tail", "n": 24, "timeout": 14400},
        {"name": "levelC_windows", "cmd": "python -m paper_results.reproduce estranet --suites ches_windows", "show": "tail", "n": 24, "timeout": 21600},
        {"name": "levelC_figure2", "cmd": "python -m bi_scope_certificates.prepare_aesrd_estranet_h5 && for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds --output-dir results/paper_results/recomputed/selected_centers/$ds; done && python -m paper_results.reproduce report", "show": "tail", "n": 24, "timeout": 14400},
        {"name": "levelD", "cmd": "python -m paper_results.reproduce fullversion", "show": "tail", "n": 24, "timeout": 14400},
    ],
    "02_bayes_oracle": [
        {"name": "help", "cmd": "python -m bayes_oracle.gaussian_oracle_suite --help"},
        {"name": "run", "cmd": "python -m bayes_oracle.gaussian_oracle_suite --amplitude 3.0 --sigma 1.0 --n-train 102400 --n-attack 100000 --n-seeds 3 --output-dir results/bayes_gaussian_oracle_suite"},
        {"name": "files", "cmd": "ls results/bayes_gaussian_oracle_suite"},
        {"name": "summary", "cmd": "python -c \"import pandas as pd; print(pd.read_csv('results/bayes_gaussian_oracle_suite/gaussian_oracle_suite_summary.csv')[['model', 'p_bayes_mean', 'p_suite_mean', 'p_suite_plus_mean']].to_string(index=False))\""},
    ],
    "03_bi_suite_8datasets": [
        {"name": "eval_ascad_d0", "cmd": f"{T}; python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split --dataset ascad_desync_0 --split attack --models-dir $T/models/pretrained_models/models --dataset-dir $T/datasets --tches20-src-dir $T/src --output-dir results/bi_suite_8datasets/tches20_fixed_split", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "outputs", "cmd": "head -5 results/bi_suite_8datasets/tches20_fixed_split/ascad_desync_0_attack_success.csv && cat results/bi_suite_8datasets/tches20_fixed_split/ascad_desync_0_attack_summary.csv"},
        {"name": "estranet_help", "cmd": "python -m bi_suite_8datasets.evaluate_estranet_bi_suite --help", "show": "tail", "n": 24},
        {"name": "estranet_suite", "cmd": "python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 --skip-selected --out-dir results/bi_suite_8datasets/estranet", "show": "tail", "n": 24, "timeout": 7200},
    ],
    "04_bi_scope_certificates": [
        {"name": "full_suite", "cmd": "python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset ascad_desync_50 --suite-csv results/paper_results/bi_suite_dataset_summary.csv --output-dir results/bi_scope_certificates/full_suite_loc", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "full_suite_summary", "cmd": "cat results/bi_scope_certificates/full_suite_loc/ascad_desync_50_full_suite_bi_loc_summary.csv"},
        {"name": "profile", "cmd": "python -m bi_scope_certificates.profile_epsilon_calibration --dataset ascad_desync_50 --public-only --output-dir results/bi_scope_certificates/calibration/ascad_desync_50", "show": "tail", "n": 24, "timeout": 3600},
        {"name": "profile_outputs", "cmd": "ls results/bi_scope_certificates/calibration/ascad_desync_50 && head -12 results/bi_scope_certificates/calibration/ascad_desync_50/epsilon_variant_distances.csv"},
    ],
    "05_epi_comparison": [
        {"name": "head_to_head", "cmd": f"{T}; python -m epi_comparison.epi_head_to_head --dataset ascad_desync_0 --dataset-dir $T/datasets --tches20-src-dir $T/src --models-dir $T/models --output-dir results/epi_comparison/main", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "temperature", "cmd": "cat results/epi_comparison/main/ascad_desync_0/temperature_stress.csv"},
        {"name": "summary", "cmd": "python -m epi_comparison.summarize_epi_head_to_head --input-dir results/paper_results/recomputed/epi_comparison/main --output-dir results/epi_comparison/summary && cat results/epi_comparison/summary/summary.csv", "timeout": 1800},
    ],
    "06_multibyte_composition": [
        {"name": "fetch", "cmd": "python -m multibyte_composition.fetch_raw_ascad", "timeout": 14400},
        {"name": "scan", "cmd": "python -m multibyte_composition.scan_raw_ascad_pois", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "train_byte0", "cmd": "python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte 0", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "train_rest", "cmd": "for b in 1 2 3 4 5 8 15; do python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b; done", "show": "tail", "n": 24, "timeout": 14400},
        {"name": "compose", "cmd": "python -m multibyte_composition.compose_four_byte_recovery --campaigns 250 --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000", "show": "tail", "n": 24, "timeout": 14400},
        {"name": "summary", "cmd": "python -c \"import pandas as pd; print(pd.read_csv('results/multibyte_composition/byte_attackers/multibyte_composition_summary.csv')[['trace_count', 'campaigns', 'full_successes', 'empirical_full_success', 'full_kl_binom_upper', 'median_full_rank']].to_string(index=False))\""},
    ],
    "07_dimension_stability": [
        {"name": "manifest", "cmd": "python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only --dimensions 100 --seeds 0 --metrics GKOV_MI,MLP_PI --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability && cat results/dimension_stability/manifest.csv"},
        {"name": "task0", "cmd": "python -m dimension_stability.ches_nonbi_metric_cell --task-id 0 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "task1", "cmd": "python -m dimension_stability.ches_nonbi_metric_cell --task-id 1 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability && cat results/dimension_stability/ches_nonbi_highdim_checkpoint.csv", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "synthetic", "cmd": "python -m dimension_stability.multivariate_stability_driver --dataset synthetic --sweep-dimensions --dimensions 10,20,50 --n-seeds 2 --no-baselines --output-dir results/dimension_stability/synthetic && ls results/dimension_stability/synthetic", "show": "tail", "n": 24, "timeout": 7200},
    ],
    "08_locality_diagnostics": [
        {"name": "sweep", "cmd": f"{T}; python -m locality_diagnostics.ascad_temporal_window_sweep --ascad-h5 $T/datasets/ASCAD_dataset/ASCAD.h5 --cnn-model $T/models/pretrained_models/models/zaid_ascad_desync_0_no_preprocessing_0.hdf5 --tau-values 50,700 --n-seeds 1 --strategy center --output-dir results/locality_diagnostics", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "outputs", "cmd": "cat results/locality_diagnostics/tau_sweep_results.csv && ls results/locality_diagnostics/figures"},
    ],
    "09_attacker_scope_diagnostics": [
        {"name": "cell", "cmd": f"{T}; python -m attacker_scope_diagnostics.scope_cell_runner --dataset ascad_desync_0 --seed 0 --scope 1 --max-features 50 --dataset-dir $T/datasets --tches20-src-dir $T/src --device cpu --output-dir results/attacker_scope_diagnostic --checkpoint results/attacker_scope_diagnostic/scope_checkpoint.csv", "show": "tail", "n": 24, "timeout": 14400},
        {"name": "outputs", "cmd": "cat results/attacker_scope_diagnostic/scope_checkpoint.csv"},
    ],
    "10_model_training": [
        {"name": "mlp", "cmd": f"{T}; python -m model_training.train_mlp_example --dataset ascad_desync_0 --data-dir $T/datasets --tches20-src-dir $T/src", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "cnn", "cmd": f"{T}; python -m model_training.train_cnn_example --dataset ascad_desync_0 --data-dir $T/datasets --tches20-src-dir $T/src --epochs 3", "show": "tail", "n": 24, "timeout": 7200},
        {"name": "outputs", "cmd": "ls -R checkpoints/model_training && cat checkpoints/model_training/ascad_desync_0/mlp/metadata.json"},
    ],
    "11_core_datasets_scripts": [
        {"name": "data_loader", "cmd": "python -m core.data_loader"},
        {"name": "setup_datasets", "cmd": "python -m datasets.setup_datasets --info; python -m datasets.setup_datasets --verify --data-dir datasets/raw", "stop_on_error": False},
        {"name": "plot_help", "cmd": "python -m scripts.plot_bi_loc_radius_curves --help"},
    ],
}
