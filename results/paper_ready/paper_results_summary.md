# Frozen Results Digest

All values below come from accepted frozen JSON artifacts. Three-seed summaries describe consistency and robustness; no inferential test is reported. Reference-grid canonical validation is development validation, and industrial canonical validation is external industrial validation.

## 1. Temporal dominance

The GRU anchor achieved development-validation node-macro MAE 0.00034479027031545902, while `temporal_residual_only` achieved 0.00032925806226992394, a relative reduction of 4.5048278280370821%. Graph WaveNet achieved 0.00033290577237537751 on the same four-grid macro endpoint.

## 2. Physical/spatial contribution

Against temporal-only MAE 0.00032925806226992394, uniform spatial utility, utility-prior without physics, and full PUC-RSTAttn V2 yielded 0.00033166683391395405, 0.00033227490340447452, and 0.0003317687407702807, respectively. These exact ablations support a weak/inconsistent spatial contribution on this endpoint.

## 3. Reference-grid 25%

BTD-FL achieved 0.00034577673225562013 +/- 2.5159437387379291e-06 development-validation node-macro MAE, versus Local 0.00037393045689658309 +/- 2.2930193219281338e-06, FedAvg 0.0003613628958204597 +/- 3.9633006414039016e-06, and FedFomo-style 0.00036882159763834764 +/- 6.2437679238411986e-06. Relative improvements were 7.5292762397661512%, 4.2968649089520836%, and 6.2292889579418977%, with 3/3, 3/3, and 3/3 seed wins.

## 4. Reference-grid 50%

BTD-FL achieved 0.0003354481131077414 +/- 1.7520254865545847e-06, versus Local 0.00035062209210361155 +/- 8.7581047850821335e-07, FedAvg 0.00035730080002285055 +/- 4.0490591507433749e-06, and FedFomo-style 0.00034906129802434702 +/- 6.4150386537916801e-07. Relative improvements were 4.3273086366926661%, 6.1008541313320022%, and 3.9003276211258986%, with 3/3, 3/3, and 3/3 seed wins.

## 5. FedFomo-style mechanism comparison

BTD-FL reduced validation node-macro MAE relative to FedFomo-style by 6.2292889579418977% at 25% and 3.9003276211258986% at 50%, winning in 3/3 and 3/3 seeds.

## 6. BTD mechanism ablations

At seed 42 and 25% history, final benefit-selected raw temporal transfer achieved validation MAE 0.0003437680937597074; removing benefit selection gave 0.00034831629772866598; transferring the full trainable donor state gave 0.00034676604563052874; and the historical benefit-selected formulation that reused the target-adapted probe state gave 0.00034481940069838762. This separates WHO is selected, WHAT is transferred, and whether adapted probe state is reused.

## 7. Industrial external 25%

On hourly industrial load/energy measurements, BTD-FL achieved 3.7056023919283034 +/- 0.098772400332740679 external-validation node-macro MAE, versus Industrial Local 4.5470416467404426 +/- 0.0084173251555798786. The relative improvement was 18.507651835077201% with 3/3 wins. Versus FedFomo-style, it was 25.091363559561923% with 3/3 wins.

## 8. Industrial external 50%

BTD-FL achieved 3.4125324533340362 +/- 0.057019394103398083, versus Industrial Local 3.9625684785156601 +/- 0.13424253160853794. The relative improvement was 13.822819658933797% with 3/3 wins. Versus FedFomo-style, it was 24.936022167924456% with 3/3 wins. Node WAPE and sMAPE were not uniformly improved.

## 9. Negative-transfer evidence

At 25%, industrial FedAvg and FedProx validation MAE were 9.5015906434319586 and 12.29245878993882, both above Industrial Local 4.5470416467404426. At 50%, they were 8.4582056799063547 and 11.272902497597004, versus Local 3.9625684785156601. In addition, the industrial 25% benefit graph contained 1 non-positive directed edge; the reference 50% graph contained 1.

## 10. Benefit/fallback behavior

Positive/non-positive/fallback counts were 36/0/0 for reference 25%, 35/1/0 for reference 50%, 11/1/0 for industrial 25%, and 12/0/0 for industrial 50%. The fallback was never triggered in these accepted runs, so fallback performance is not empirically established.

Canonical TEST remains locked; no TEST result is reported.
