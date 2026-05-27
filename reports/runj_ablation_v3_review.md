# Run J ablation — deep review

For each branch: paired baseline (Run F) vs variant (Run J branch enabled).
Wins = variant fixed; Losses = variant broke. Multi-box >1 is a regression flag.

## olympiad

  n=19    baseline=2/19 (10.5%)    variant=2/19 (10.5%)    Δ=+0.0pp
  wins(v_only)=0  losses(b_only)=0  flips_total=0
  baseline len: mean=23967 p50=22983 p95=64114
  variant  len: mean=23563 p50=13902 p95=56198
  multi-box (>1 \boxed{}): baseline=10, variant=13

## trig

  n=40    baseline=21/40 (52.5%)    variant=20/40 (50.0%)    Δ=-2.5pp
  wins(v_only)=1  losses(b_only)=2  flips_total=3
  baseline len: mean=8907 p50=6692 p95=22569
  variant  len: mean=8944 p50=7042 p95=23960
  multi-box (>1 \boxed{}): baseline=25, variant=24

  Wins (1 samples — variant fixed):
    [646] Find a "reasonable" upper-bound on the error in approximating $f(x) = (x-1) \cdot \ln(x-1)$ by its 3rd order Taylor polynomial $P_{3}(x)$ about $a=2$ valid for all values of $x$ such that $|x-2| \le 0
      gold: "['0.650277']"
      variant_extracted: '0.650277'

  Losses (2 samples — variant broke):
    [354] For each of the following, select the most appropriate unit of measurement: The mass of a booger uses [ANS] The mass of a pair of earrings uses [ANS] The weight of an passenger airplane uses [ANS] The
      gold: "['11 + 4*cos( (4/(2*11) +1)*pi)', '(11 + 4*cos( (4/(2*11) +1)*pi) )*cos( (4/(2*1"
      baseline_extracted: "['11+4\\\\cos((13\\\\pi)/(11))', '(11+4\\\\cos((13\\\\pi)/(11)))\\\\cos((13\\\\pi)/(11))', '"  ✓
      variant_extracted:  "['7.634986', '-6.421031', '-4.127801', '0']"  ✗
    [457] Find the sum of all positive integers $x$ such that there exists integers $a$ and $b$ that satisfy $$$|x^2 - 92x + 2099| = 2^a3^b - 8.$$$
      gold: "['85.9436692696235']"
      baseline_extracted: '(270)/(\\pi)'  ✓
      variant_extracted:  '85.943689'  ✗

## geometry

  n=83    baseline=40/83 (48.2%)    variant=39/83 (47.0%)    Δ=-1.2pp
  wins(v_only)=5  losses(b_only)=6  flips_total=11
  baseline len: mean=13276 p50=8778 p95=41774
  variant  len: mean=13362 p50=9833 p95=39016
  multi-box (>1 \boxed{}): baseline=50, variant=47

  Wins (3 samples — variant fixed):
    [217] Suppose $b > 1$ is a real number where $\log_5 (\log_5 b + \log_b 125) = 2$ . Find $log_5 \left(b^{\log_5 b}\right) + log_b \left(125^{\log_b 125}\right).$
      gold: "['2.55', '4.41672955930064']"
      variant_extracted: "['2.55', '2.55\\\\sqrt{3}']"
    [305] We now define an algorithm: The definition of a(n) is: Let n be a positive integer. For each prime divisor p of n, consider the highest power of p which does not exceed n. The sum a(n) of these powers
      gold: "['0.875', '-0.75']"
      variant_extracted: "['0.875', '-0.75']"
    [635] Suppose $Q=37.6(0.746)^t$. Give the starting value $a$, the growth factor $b$, and the growth rate $r$ if $Q=a \cdot b^t=a(1+r)^t$. $a=$ [ANS] $b=$ [ANS] $r=$ [ANS] \%
      gold: "['0.0403333333333333*x +15', '2107.43801652893']"
      variant_extracted: "['f(x)=15+(4.84)/(120)x', '2107.438']"

  Losses (3 samples — variant broke):
    [75] Mike can be paid in one of two ways based on the amount of merchandise he sells: Plan A: A salary of $\$1{,}100.00$ per month, plus a commission of $11\%$ of sales, OR Plan B: A salary of $\$1{,}450.0
      gold: "['13^2 + (x-4)^2 = x^2', '23.125']"
      baseline_extracted: "['169+(x-4)^2=x^2', '23.125']"  ✓
      variant_extracted:  "['x^2=(x-4)^2+169', '23.125']"  ✗
    [379] A company produce snowboards. Fixed costs are \$1320 and the cost per snowboard is \$260. An order has been placed for 6 snowboards. What should the retail price be in order for the company to break e
      gold: "['7.95110055275369']"
      baseline_extracted: '\\sqrt{63.22}'  ✓
      variant_extracted:  '7.951099'  ✗
    [483] Let $x,y$ be two non-negative real numbers such that $y\sqrt{2016-x^2} + x\sqrt{2016-y^2} = 2016$ . Then, the maximum possible value of $x+3y$ can be expressed as $m\sqrt{n}$ , where $n$ is not divisi
      gold: "['4.10841829841869']"
      baseline_extracted: '(555)/(43\\pi)'  ✓
      variant_extracted:  '4.1084'  ✗

## stats_hyp_test

  n=59    baseline=19/59 (32.2%)    variant=22/59 (37.3%)    Δ=+5.1pp
  wins(v_only)=3  losses(b_only)=0  flips_total=3
  baseline len: mean=19217 p50=18184 p95=37227
  variant  len: mean=20277 p50=19508 p95=37714
  multi-box (>1 \boxed{}): baseline=51, variant=52

  Wins (3 samples — variant fixed):
    [193] A particle move along x-axis and displacement varies with time $t$ as $x=(t^3-3t^2-9t+5)$. Then
      gold: "['2.30357142857143']"
      variant_extracted: '2.303571'
    [365] Use a graphing calculator to decide which viewing rectangle (A)-(D) produces the most appropriate graph of the equation. y=\sqrt[4]{1296-x^2} Choose one: [ANS] A. [-10,10] by [-2,8]  B. [-10,10] by [-
      gold: "['A', 'C']"
      variant_extracted: "['A', 'C']"
    [691] How many rounds of golf do those physicians who play golf play per year? A survey of 12 physicians revealed the following numbers: 8, \quad 42, \quad 16, \quad 3, \quad 32, \quad 37, \quad 20, \quad 1
      gold: "['A', 'C']"
      variant_extracted: "['A', 'C']"

## stats_regression

  n=27    baseline=12/27 (44.4%)    variant=12/27 (44.4%)    Δ=+0.0pp
  wins(v_only)=1  losses(b_only)=1  flips_total=2
  baseline len: mean=16364 p50=13936 p95=38676
  variant  len: mean=15403 p50=11182 p95=35859
  multi-box (>1 \boxed{}): baseline=19, variant=14

  Wins (1 samples — variant fixed):
    [892] An aerial photograph from a U-2 spy plane is taken of a building suspected of housing nuclear warheads. When the photograph is taken, the angle of elevation of the sun is $40 ^ \circ$. By comparing th
      gold: "['1.6105+0.8574*x', '32', 'Extrapolation', 'Interpolation', 'Extrapolation', 'In"
      variant_extracted: "['32', '0.8574x+1.6105,32,extrapolation,interpolation,extrapolation,interpolatio"

  Losses (1 samples — variant broke):
    [444] Find the quotient and remainder in each of the following. $80\vert\overline{9075}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $77\vert\overline{8685}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $56\vert\o
      gold: "['-0.203125', '4.890625', '-4.0625']"
      baseline_extracted: "['-0.203125', '4.890625', '-4.0625']"  ✓
      variant_extracted:  "['0.17307692307692307', '3.6115384615384616', '-3.7615384615384615']"  ✗

## stats_descriptive

  n=49    baseline=21/49 (42.9%)    variant=24/49 (49.0%)    Δ=+6.1pp
  wins(v_only)=3  losses(b_only)=0  flips_total=3
  baseline len: mean=14693 p50=12509 p95=36152
  variant  len: mean=16879 p50=15503 p95=35456
  multi-box (>1 \boxed{}): baseline=33, variant=37

  Wins (3 samples — variant fixed):
    [362] Factor the difference of squares: $36x^{2}-25y^{2}=$ [ANS]
      gold: "['A', 'E', 'A', 'A', 'A']"
      variant_extracted: "['A', 'E', 'A', 'A', 'A']"
    [779] Given the function $f(x)=\begin{cases}\sin x,&\text{if } \cos x \text{ is a rational number,}\\\cos^2x,&\text{if } \cos x \text{ is an irrational number,}\end{cases}$ find the value of the integral $\
      gold: "['80', '76', '77', '78.5', '81', '78', '79.5', '82', '80.5', '83', '84.5', '80',"
      variant_extracted: "['80', '76,77,78.5,81,78,79.5,82,80.5,83,84.5', '80', 'A', '80,76,77,78.5,81,78,"
    [908] A typical cup of coffee contains about 100 mg of caffeine and every hour approximately 17\% is metabolized and eliminated. (a) Write $C$, the amount of caffeine in the body in mg as a function of $t$,
      gold: "['0.5', '0', '0.25', '1.25', 'Biology']"
      variant_extracted: "['0.5', '0', '0.25', '1.25', 'Biology']"

## calculus

  n=7    baseline=2/7 (28.6%)    variant=3/7 (42.9%)    Δ=+14.3pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=18873 p50=13220 p95=42394
  variant  len: mean=17555 p50=13248 p95=40406
  multi-box (>1 \boxed{}): baseline=5, variant=5

  Wins (1 samples — variant fixed):
    [498] A list of positive integers satisfies the following properties: (A) The mean of the list is $8$. (2) The median of the list is $13$. (D) The mode of the list is $15$.  Moreover, the range of the list
      gold: "['49', '44.5', '44.05', '44.005', '44']"
      variant_extracted: "['49', '44.5', '44.05', '44.005', '44']"

## prob_combi

  n=23    baseline=12/23 (52.2%)    variant=13/23 (56.5%)    Δ=+4.3pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=16824 p50=13958 p95=49100
  variant  len: mean=15701 p50=10141 p95=47561
  multi-box (>1 \boxed{}): baseline=15, variant=15

  Wins (1 samples — variant fixed):
    [884] Given $f(x)=x^2$, after performing the following transformations: shift upward 74 units and shift 59 units to the right, the new function $g(x)=$ [ANS]
      gold: "['2.75 x + 40108', '4.86 x', '4.86 x -(2.75 x + 40108)', '19009']"
      variant_extracted: "['40108+2.75x', '4.86x', '2.11x-40108', '19009']"

## number_alg

  n=6    baseline=3/6 (50.0%)    variant=5/6 (83.3%)    Δ=+33.3pp
  wins(v_only)=2  losses(b_only)=0  flips_total=2
  baseline len: mean=17094 p50=16329 p95=50277
  variant  len: mean=14557 p50=10590 p95=49614
  multi-box (>1 \boxed{}): baseline=4, variant=3

  Wins (2 samples — variant fixed):
    [103] If the distance from the town of Bree to Weathertop is 16 miles on a 45 degree upward slope, what is the elevation gain (omit units)? [ANS]
      gold: "['(-1)^(n+1)/(n^4)', '(n+2)/(n+4)']"
      variant_extracted: "['(-1)^{n-1}/n^4', '(n+2)/(n+4)']"
    [839] Evaluate $\sum_{n=0}^\infty \mathrm{Arccot}(n^2+n+1)$, where $\mathrm{Arccot}\,t$ for $t \geq 0$ denotes the number $\theta$ in the interval $0 < \theta \leq \pi/2$ with $\cot \theta = t$.
      gold: '998'
      variant_extracted: '998'
