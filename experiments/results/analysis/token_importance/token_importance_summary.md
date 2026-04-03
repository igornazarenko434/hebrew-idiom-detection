# Token Importance Analysis (Mission 3.1)

Generated: 2026-01-06 15:15

## Selected Best Models
- seen_test | cls | hebrew: neodictabert (seed 42, F1=0.9722)
- seen_test | cls | multilingual: xlm-roberta-base (seed 42, F1=0.9259)
- seen_test | span | hebrew: neodictabert (seed 456, F1=0.9977)
- seen_test | span | multilingual: bert-base-multilingual-cased (seed 456, F1=0.9965)
- unseen_test | cls | hebrew: neodictabert (seed 456, F1=0.9270)
- unseen_test | cls | multilingual: xlm-roberta-base (seed 456, F1=0.9125)
- unseen_test | span | hebrew: dictabert (seed 123, F1=0.7901)
- unseen_test | span | multilingual: xlm-roberta-base (seed 456, F1=0.6450)

## Attribution Examples
- seen_test | cls | hebrew: neodictabert seed 42 (F1=0.9722) → token_importance_neodictabert_cls_seen_test.json
- seen_test | cls | multilingual: xlm-roberta-base seed 42 (F1=0.9259) → token_importance_xlm-roberta-base_cls_seen_test.json
- seen_test | span | hebrew: neodictabert seed 456 (F1=0.9977) → token_importance_neodictabert_span_seen_test.json
- seen_test | span | multilingual: bert-base-multilingual-cased seed 456 (F1=0.9965) → token_importance_bert-base-multilingual-cased_span_seen_test.json
- unseen_test | cls | hebrew: neodictabert seed 456 (F1=0.9270) → token_importance_neodictabert_cls_unseen_test.json
- unseen_test | cls | multilingual: xlm-roberta-base seed 456 (F1=0.9125) → token_importance_xlm-roberta-base_cls_unseen_test.json
- unseen_test | span | hebrew: dictabert seed 123 (F1=0.7901) → token_importance_dictabert_span_unseen_test.json
- unseen_test | span | multilingual: xlm-roberta-base seed 456 (F1=0.6450) → token_importance_xlm-roberta-base_span_unseen_test.json