# BI Scope Certificate Lane

This lane reorganizes the current evidence around three declared scopes:

- `BI^sing`: one frozen attacker per dataset.
- `BI^suite`: the existing finite frozen-suite table.
- `BI^loc`: score-local families around frozen centers, evaluated by relaxed
  margin counts.

The local-family job deliberately avoids the older unrestricted `BIarch`
wording.  It defines local scope directly in normalized score space:

```text
sup_y ||bar_s_h(y) - bar_s_center(y)||_infty <= epsilon
```

and counts attack traces satisfying `rho_center >= -2 epsilon`.

Main entry points:

```bash
python -m bi_scope_certificates.compute_bi_loc_margin_curves --help
python -m bi_scope_certificates.compute_full_suite_bi_loc --help
python -m bi_scope_certificates.build_bi_scope_tables --help
```

Outputs should be written under `results/bi_scope_certificates/` or another ignored output path.
