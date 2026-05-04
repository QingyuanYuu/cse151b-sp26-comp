# Error Analysis Report

_Source: `results/baseline_v1_val.jsonl`_

**Overall accuracy**: 62.67% (141/225)

## By question type

| key | n | correct | acc |
|---|---|---|---|
| free_multi | 83 | 42 | 50.60% |
| mcq | 75 | 55 | 73.33% |
| free_single | 67 | 44 | 65.67% |

## By topic

| key | n | correct | acc |
|---|---|---|---|
| other | 157 | 102 | 64.97% |
| geometry | 21 | 12 | 57.14% |
| probability | 16 | 8 | 50.00% |
| series | 13 | 7 | 53.85% |
| calculus | 10 | 8 | 80.00% |
| linalg | 6 | 3 | 50.00% |
| ode | 1 | 1 | 100.00% |
| complex | 1 | 0 | 0.00% |

## By question length

| key | n | correct | acc |
|---|---|---|---|
| medium(150-500) | 119 | 78 | 65.55% |
| short(<150) | 59 | 44 | 74.58% |
| long(500-1500) | 39 | 17 | 43.59% |
| vlong(>=1500) | 8 | 2 | 25.00% |

## Failure modes

| mode | count |
|---|---|
| no_box | 0 |
| truncated | 5 |
| wrong_shape | 9 |
| numeric_but_wrong | 70 |
| other | 0 |

**No-box response lengths**: n=6, median=43743, max=63747

## Top confident-but-wrong (boxed something, graded wrong)

- id=13 type=mcq topic=calculus boxed=['D'] gold='J'
- id=22 type=free_multi topic=other boxed=['1.600000, 1.760000, 0.16 \\lceil p/16 \\rceil, up, 16, 0.16'] gold=['1.6', '1.76', '0.16*p/16', 'up', '1', '0.16']
- id=29 type=free_multi topic=other boxed=['332.143'] gold=['442.857142857143', '332.142857142857']
- id=39 type=free_single topic=other boxed=['2.3'] gold=['2.2892']
- id=70 type=free_single topic=geometry boxed=['4.64258'] gold=['4.64257581030492']
- id=83 type=free_multi topic=other boxed=['0'] gold=['yes', 'yes', 'no']
- id=93 type=free_multi topic=other boxed=['90/(4 - \\sin \\theta)'] gold=['20/[5+3*cos(t)]', '90/[4-sin(t)]']
- id=123 type=mcq topic=series boxed=['D'] gold='B'
- id=125 type=free_single topic=probability boxed=['13'] gold=['12.3941512253475']
- id=132 type=free_multi topic=other boxed=['7.34000'] gold=['C', 'A']
