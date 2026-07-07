# imajin — figure gallery

Every figure type imajin currently produces, rendered at **default styling on synthetic
data** — a reference for the plotting tools' look, colour palette and options. An
interactive version is [`index.html`](index.html) (open it in a browser).

---

## Palette & typography

Control is a de-emphasised slate grey (`#636867`); conditions get colour — colorblind-safe
(min ΔE ≥ 15.2). Fonts: **Noto Sans** (default) / **Noto Serif**. Axis, tick and legend
labels are auto-formatted to scientific typography (`Mean Intensity`, `µm²`, `(A.U.)`).

<img src="00_palette.png" width="660"><br>
<sub>6-colour categorical palette — verified under normal / deuteranopia / protanopia / grayscale.</sub>

<img src="00_serif.png" width="380"><br>
<sub><code>font="serif"</code> — Noto Serif option (default is Noto Sans).</sub>

## Group comparison — `plot_group_distribution`

Chart `kind`, paired lines, post-hoc brackets and the condition matrix are all options of
one tool.

<table>
<tr>
<td align="center"><img src="01_dist_box.png" width="320"><br><sub><code>kind="box"</code> — box + points + mean ± 95% CI (default)</sub></td>
<td align="center"><img src="01_dist_bar.png" width="320"><br><sub><code>kind="bar"</code> — mean + SEM bars + points</sub></td>
</tr>
<tr>
<td align="center"><img src="01_dist_violin.png" width="320"><br><sub><code>kind="violin"</code> — density + points</sub></td>
<td align="center"><img src="01_dist_dots.png" width="320"><br><sub><code>kind="dots"</code> — all points + mean ± SEM (best for small n)</sub></td>
</tr>
<tr>
<td align="center"><img src="02_posthoc.png" width="320"><br><sub>3 groups — multiplicity-corrected post-hoc brackets</sub></td>
<td align="center"><img src="03_paired.png" width="320"><br><sub><code>paired=True</code> — within-subject connecting lines</sub></td>
</tr>
</table>

## Condition matrix — `condition_matrix=`

Filled ● (positive) / open ○ (negative) circles per factor beneath the bars — replaces
long compound tick labels; reusable across plots.

<img src="08_condition_matrix.png" width="440"><br>
<sub><code>plot_group_distribution</code> — Treatment / Genotype condition table (columns = bars).</sub>

<img src="07_grouped_bars.png" width="470"><br>
<sub><code>plot_grouped_bars</code> — grouped bars + condition table (Activation) + circle legend + per-condition significance.</sub>

## Time series — `plot_timecourse`

<img src="04_timecourse.png" width="470"><br>
<sub>Group mean ± SEM band over individual traces.</sub>

## Correlation — `plot_scatter`

<img src="05_scatter.png" width="420"><br>
<sub>Grouped colour + regression line + <code>r</code> / <code>p</code>.</sub>

## Calcium imaging — `plot_dff_heatmap`

<img src="06_heatmap.png" width="440"><br>
<sub>Cell × time ΔF/F₀ raster.</sub>
