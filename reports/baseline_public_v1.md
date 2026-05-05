# Error Analysis Report

_Source: `results/baseline_public_v1.enriched.jsonl`_

**Overall accuracy**: 60.12% (677/1126)

## By question type

| key | n | correct | acc |
|---|---|---|---|
| free_multi | 414 | 195 | 47.10% |
| mcq | 375 | 275 | 73.33% |
| free_single | 337 | 207 | 61.42% |

## By topic

| key | n | correct | acc |
|---|---|---|---|
| other | 731 | 443 | 60.60% |
| geometry | 123 | 60 | 48.78% |
| calculus | 81 | 67 | 82.72% |
| probability | 80 | 38 | 47.50% |
| series | 68 | 41 | 60.29% |
| linalg | 22 | 15 | 68.18% |
| ode | 15 | 9 | 60.00% |
| complex | 6 | 4 | 66.67% |

## By question length

| key | n | correct | acc |
|---|---|---|---|
| medium(150-500) | 588 | 363 | 61.73% |
| short(<150) | 311 | 239 | 76.85% |
| long(500-1500) | 202 | 69 | 34.16% |
| vlong(>=1500) | 25 | 6 | 24.00% |

## Failure modes

| mode | count |
|---|---|
| no_box | 4 |
| truncated | 35 |
| wrong_shape | 70 |
| numeric_but_wrong | 340 |
| other | 0 |

**No-box response lengths**: n=40, median=40056, max=63747

## Top confident-but-wrong (boxed something, graded wrong)

- id=1 type=mcq topic=other boxed=['E'] gold='F'
- id=2 type=free_multi topic=other boxed=['2.32752'] gold=['143.224229233795', '2.32624773420025']
- id=8 type=free_single topic=other boxed=['0.447061'] gold=['(1/2)^[(1999-1963)/31]']
- id=12 type=free_multi topic=probability boxed=['380, 315, 14, 310'] gold=['380', '315', '13', '310']
- id=13 type=mcq topic=calculus boxed=['D'] gold='J'
- id=16 type=free_multi topic=geometry boxed=['3.142'] gold=['atan(4.76)', 'pi']
- id=21 type=free_multi topic=other boxed=['70t^4(1-t)^4'] gold=['3*t^1*(1-t)^2', '6*t^2*(1-t)^2', '10*t^3*(1-t)^2', '7*t^6*(1-t)^1', '70*t^4*(1-t)^4']
- id=22 type=free_multi topic=other boxed=['1.600000, 1.760000, 0.16 \\lceil p/16 \\rceil, up, 16, 0.16'] gold=['1.6', '1.76', '0.16*p/16', 'up', '1', '0.16']
- id=23 type=free_multi topic=other boxed=['2,1,2,1'] gold=['B', 'A', 'B', 'A']
- id=24 type=mcq topic=calculus boxed=['E'] gold='A'
