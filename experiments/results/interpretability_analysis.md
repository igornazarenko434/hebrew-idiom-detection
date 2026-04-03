# Interpretability Analysis (Mission 6.1)

Generated: 2026-01-06 15:20

## Selection Criteria
- High-confidence errors
- Low-confidence correct
- Frequent misclassified idioms
- Target expressions per split/task:
  - seen_test | cls: עשה סצנה, ירה לכל הכיוונים, קיפל את הזנב, החזיק אצבעות, קבר את עצמו, ירד לו האסימון
  - seen_test | span: הרים את הראש, ירד לו האסימון, הניף דגל לבן, נתן גז, נכנס מתחת לאלונקה, משך בחוטים
  - unseen_test | cls: רץ אחרי הזנב של עצמו, נשאר מאחור, חצה קו אדום, שבר שתיקה, חתך פינה, איבד את הראש
  - unseen_test | span: רץ אחרי הזנב של עצמו, נשאר מאחור, חצה קו אדום, שבר שתיקה, חתך פינה, איבד את הראש

## Files
- Selected cases: experiments/results/analysis/token_importance/selected_cases.csv
- Token importance summary: experiments/results/analysis/token_importance/token_importance_summary.md