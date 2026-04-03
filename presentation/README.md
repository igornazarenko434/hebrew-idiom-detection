# Hebrew Idiom Detection - Presentation

A self-contained web-based presentation for the final project:
**"Detection Without Localization: Compositional Generalization Failures in Transformer Models for Hebrew Idiom Boundaries"**

## How to Run

1. Open `index.html` in any modern browser (Chrome, Firefox, Safari, Edge)
2. No build step, no npm, no internet required
3. Works best in fullscreen (press `F`)

## Presenter Controls

| Key | Action |
|-----|--------|
| `→` / `Space` / `PageDown` | Next slide |
| `←` / `PageUp` | Previous slide |
| `Home` | First slide |
| `End` | Last slide |
| `N` | Toggle speaker notes panel |
| `F` | Toggle fullscreen |
| Click right half | Next slide |
| Click left half | Previous slide |
| Swipe left/right | Navigate (touch devices) |

## Where to Edit Content

- **Opening words (Slide 2):** Search for `[TODO: Insert student opening words here` in `index.html`
- **All slide content:** Each slide is a `<div class="slide">` block in `index.html`
- **Speaker notes:** Edit the `data-notes="..."` attribute on each slide div
- **Figures:** Located in `assets/figures/` - replace any PNG to update visuals
- **Styling:** All CSS is embedded at the top of `index.html`

## File Structure

```
presentation/
├── index.html              # Self-contained presentation (HTML + CSS + JS)
├── README.md               # This file
└── assets/
    └── figures/
        ├── error_distribution_span_aggregated.png
        ├── seen_unseen_comparison.png
        ├── error_heatmap_span.png
        ├── generalization_bar_cls.png
        ├── generalization_bar_span.png
        ├── per_idiom_heatmap_span_unseen_test.png
        ├── neodictabert_tsne_seen_vs_unseen.png
        ├── neodictabert_attention_ratio_correct_vs_partial_end.png
        ├── learning_curves_span.png
        ├── model_comparison_cls_seen.png
        ├── model_comparison_span_unseen.png
        └── interp_partial_end_example.png
```

## Content Map

| Slide | Title | LaTeX Source |
|-------|-------|-------------|
| 1 | Title | Paper title, authors |
| 2 | Opening Words: Plan vs Reality | initial_plan.docx comparison |
| 3 | Why Hebrew Idioms? | §1 Introduction |
| 4 | Research Contributions | §1 Contributions list |
| 5 | Section Divider: Dataset | - |
| 6 | Dataset Overview | §3 Dataset, Table 1, professor_review/README |
| 7 | Data Construction Process | §3.1-3.2, professor_review/README §9 |
| 8 | Splits & Annotation Format | §3.3-3.6, Appendix H |
| 9 | Section Divider: Experiments | - |
| 10 | Models & Setup | §4.2-4.3, Appendix D |
| 11 | Main Results | §5.1, Table 1 (combined results) |
| 12 | Generalization Gap | §5.2-5.5, generalization figures |
| 13 | Error Analysis | §5.3-5.6, error distribution figures |
| 14 | Fine-Tuning vs Prompting | §5.8, Table 2 (prompting bias) |
| 15 | Interpretability | §5.7-5.8, attention + embedding figures |
| 16 | Limitations | Limitations section |
| 17 | Claims vs Evidence | Cross-reference all claims |
| 18 | Takeaways & Future Work | §7 Conclusion |

## Technical Notes

- All numbers come directly from the LaTeX paper and analysis outputs
- No numbers are invented - all sourced from `conll2026_paper.tex`
- Figures are copies from `paper/figures/` directory
- Presentation is ~16 slides, optimized for 8-12 minute talk
