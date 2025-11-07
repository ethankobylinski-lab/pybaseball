# Factor Glossary

This glossary documents the features currently implemented in the analytics
warehouse.  Each entry describes the exact calculation so analysts can audit
results and translate them to other levels of play.

## scored_first
Indicator equal to 1 when the team recorded the first scoring play of the game.
The definition respects home/away alignment using play-by-play scoring order.

## big_inning
Binary flag that becomes 1 when the team posted an inning with three or more
runs.  The threshold is configurable in SQL if different definitions are
required.

## answered_back_next_half
Binary indicator capturing whether the team immediately answered an opponent
scoring event in the very next half inning.  Useful for sequencing analysis.

## shutdown_inning_after_scoring
Rate between 0 and 1 measuring how often the team prevented runs in the half
inning directly following their own scoring half-innings.  ``NULL`` indicates
no scoring opportunities for the team in that game.

## two_out_runs / two_out_run_share
``two_out_runs`` counts runs scored with two outs. ``two_out_run_share`` divides
that value by all runs scored by the team.

## boxscore differentials
Derived from box score totals.  Includes ``hits_diff``, ``bb_diff``,
``k_diff``, ``hr_diff``, ``xbh_diff``, ``sb_diff``, ``e_diff`` and ``lob_diff``.
Positive numbers favour the team, negative numbers indicate the opponent led
in the respective category.

## Adding new factors
Register additional DuckDB views in `features/registry.py` to expose new
metrics to both the tests and the application.  Each `FeatureDefinition`
requires a unique key, view name, and description.  Once registered, the
feature is available to the `build_features` pipeline and can be surfaced in
unit tests or the `Model Builder` without further changes.
