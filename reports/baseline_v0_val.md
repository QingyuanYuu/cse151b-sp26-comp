# Error Analysis Report

_Source: `results/baseline_v0_val.enriched.jsonl`_

**Overall accuracy**: 56.44% (127/225)

## By question type

| key | n | correct | acc |
|---|---|---|---|
| free_multi | 83 | 40 | 48.19% |
| mcq | 75 | 45 | 60.00% |
| free_single | 67 | 42 | 62.69% |

## By topic

| key | n | correct | acc |
|---|---|---|---|
| other | 157 | 93 | 59.24% |
| geometry | 21 | 11 | 52.38% |
| probability | 16 | 8 | 50.00% |
| series | 13 | 4 | 30.77% |
| calculus | 10 | 7 | 70.00% |
| linalg | 6 | 3 | 50.00% |
| ode | 1 | 1 | 100.00% |
| complex | 1 | 0 | 0.00% |

## By question length

| key | n | correct | acc |
|---|---|---|---|
| medium(150-500) | 119 | 72 | 60.50% |
| short(<150) | 59 | 40 | 67.80% |
| long(500-1500) | 39 | 13 | 33.33% |
| vlong(>=1500) | 8 | 2 | 25.00% |

## Failure modes

| mode | count |
|---|---|
| no_box | 14 |
| truncated | 9 |
| wrong_shape | 22 |
| numeric_but_wrong | 53 |
| other | 0 |

**No-box response lengths**: n=25, median=28731, max=42901

## Top confident-but-wrong (boxed something, graded wrong)

- id=29 type=free_multi topic=other boxed=['442.86, 332.14'] gold=['442.857142857143', '332.142857142857']
- id=38 type=free_multi topic=other boxed=['875'] gold=['1250', '875']
- id=39 type=free_single topic=other boxed=['2.3'] gold=['2.2892']
- id=82 type=mcq topic=linalg boxed=['B'] gold='G'
- id=83 type=free_multi topic=other boxed=['Yes, No, No'] gold=['yes', 'yes', 'no']
- id=123 type=mcq topic=series boxed=['D'] gold='B'
- id=125 type=free_single topic=probability boxed=['13'] gold=['12.3941512253475']
- id=132 type=free_multi topic=other boxed=['5.06, 20.97'] gold=['C', 'A']
- id=139 type=free_multi topic=other boxed=['-\\frac{3}{4}, \\frac{3}{4}'] gold=['(-x - sqrt(9 - (y-5*b)**2))/(--4)', '(-x + sqrt(9 - (y-5*b)**2))/(--4)']
- id=250 type=free_multi topic=other boxed=['-16.81, -4.03, 6.39, B'] gold=['-16.9799', '-3.85208', '6.56392', 'B']
