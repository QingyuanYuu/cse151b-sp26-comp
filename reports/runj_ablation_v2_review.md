# Run J ablation — deep review

For each branch: paired baseline (Run F) vs variant (Run J branch enabled).
Wins = variant fixed; Losses = variant broke. Multi-box >1 is a regression flag.

## olympiad

  n=19    baseline=2/19 (10.5%)    variant=3/19 (15.8%)    Δ=+5.3pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=23967 p50=22983 p95=64114
  variant  len: mean=21625 p50=15559 p95=57762
  multi-box (>1 \boxed{}): baseline=10, variant=12

  Wins (1 samples — variant fixed):
    [641] Estimate the product by rounding each of the given values to the nearest tenth. $7.182 \times 6.385$ $\approx$ $7.2$ $\times$ [ANS]=[ANS]
      gold: '74'
      variant_extracted: '74'

## trig

  n=40    baseline=21/40 (52.5%)    variant=18/40 (45.0%)    Δ=-7.5pp
  wins(v_only)=0  losses(b_only)=3  flips_total=3
  baseline len: mean=8907 p50=6692 p95=22569
  variant  len: mean=9690 p50=8225 p95=25431
  multi-box (>1 \boxed{}): baseline=25, variant=27

  Losses (3 samples — variant broke):
    [354] For each of the following, select the most appropriate unit of measurement: The mass of a booger uses [ANS] The mass of a pair of earrings uses [ANS] The weight of an passenger airplane uses [ANS] The
      gold: "['11 + 4*cos( (4/(2*11) +1)*pi)', '(11 + 4*cos( (4/(2*11) +1)*pi) )*cos( (4/(2*1"
      baseline_extracted: "['11+4\\\\cos((13\\\\pi)/(11))', '(11+4\\\\cos((13\\\\pi)/(11)))\\\\cos((13\\\\pi)/(11))', '"  ✓
      variant_extracted:  "['7.631', '-6.427', '-4.126', '0']"  ✗
    [457] Find the sum of all positive integers $x$ such that there exists integers $a$ and $b$ that satisfy $$$|x^2 - 92x + 2099| = 2^a3^b - 8.$$$
      gold: "['85.9436692696235']"
      baseline_extracted: '(270)/(\\pi)'  ✓
      variant_extracted:  '85.94'  ✗
    [1061] Write $-8 \sin(5 t)+7 \cos(5 t)$ in the form $A \sin(B t+\phi)$ using sum or difference formulas. $-8 \sin(5 t)+7 \cos(5 t)$=[ANS]
      gold: "['sqrt(8^2+7^2)*sin(5*t+atan(-7/8)+pi)']"
      baseline_extracted: '\\sqrt{113}\\sin(5t+\\pi-\\arctan((7)/(8)))'  ✓
      variant_extracted:  'sqrt(113),5,\\pi-\\arctan((7)/(8))'  ✗

## geometry

  n=50    baseline=26/50 (52.0%)    variant=17/50 (34.0%)    Δ=-18.0pp
  wins(v_only)=0  losses(b_only)=9  flips_total=9
  baseline len: mean=13159 p50=7057 p95=48045
  variant  len: mean=13307 p50=9922 p95=43163
  multi-box (>1 \boxed{}): baseline=30, variant=32

  Losses (3 samples — variant broke):
    [379] A company produce snowboards. Fixed costs are \$1320 and the cost per snowboard is \$260. An order has been placed for 6 snowboards. What should the retail price be in order for the company to break e
      gold: "['7.95110055275369']"
      baseline_extracted: '\\sqrt{63.22}'  ✓
      variant_extracted:  '7.951'  ✗
    [445] $P$ is the probability that if you flip a fair coin, $20$ heads will occur before $19$ tails. If $P=\frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers, find the remainder when $m+n$
      gold: "['8.90117918517108', '22.6980069221863']"
      baseline_extracted: "['17\\\\pi/6', '289\\\\pi/40']"  ✓
      variant_extracted:  "['8.901', '22.698', '8.901,22.698']"  ✗
    [483] Let $x,y$ be two non-negative real numbers such that $y\sqrt{2016-x^2} + x\sqrt{2016-y^2} = 2016$ . Then, the maximum possible value of $x+3y$ can be expressed as $m\sqrt{n}$ , where $n$ is not divisi
      gold: "['4.10841829841869']"
      baseline_extracted: '(555)/(43\\pi)'  ✓
      variant_extracted:  '4.108'  ✗

## stats_hyp_test

  n=50    baseline=16/50 (32.0%)    variant=19/50 (38.0%)    Δ=+6.0pp
  wins(v_only)=3  losses(b_only)=0  flips_total=3
  baseline len: mean=19391 p50=18184 p95=37227
  variant  len: mean=18352 p50=18959 p95=33996
  multi-box (>1 \boxed{}): baseline=43, variant=39

  Wins (3 samples — variant fixed):
    [365] Use a graphing calculator to decide which viewing rectangle (A)-(D) produces the most appropriate graph of the equation. y=\sqrt[4]{1296-x^2} Choose one: [ANS] A. [-10,10] by [-2,8]  B. [-10,10] by [-
      gold: "['A', 'C']"
      variant_extracted: "['A', 'C']"
    [691] How many rounds of golf do those physicians who play golf play per year? A survey of 12 physicians revealed the following numbers: 8, \quad 42, \quad 16, \quad 3, \quad 32, \quad 37, \quad 20, \quad 1
      gold: "['A', 'C']"
      variant_extracted: "['A', 'C']"
    [1088] For each problem, select the best response. (a) Does 30 minutes of aerobic exercise each day provide significant improvement in mental performance? To investigate this issue, a researcher conducted a
      gold: "['D', 'D', 'A']"
      variant_extracted: "['D', 'D', 'A']"

## stats_regression

  n=27    baseline=12/27 (44.4%)    variant=10/27 (37.0%)    Δ=-7.4pp
  wins(v_only)=0  losses(b_only)=2  flips_total=2
  baseline len: mean=16364 p50=13936 p95=38676
  variant  len: mean=16438 p50=14785 p95=45398
  multi-box (>1 \boxed{}): baseline=19, variant=19

  Losses (2 samples — variant broke):
    [386] There are $n$ parallel lines on a plane, and there is a set $S$ of distinct points. Each point in $S$ lies on one of the $n$ lines and is colored either red or blue. Determine the minimum value of $n$
      gold: "['NEGATIVE', '0.680625']"
      baseline_extracted: "['negative', '0.680625']"  ✓
      variant_extracted:  "['negative', '0.6806']"  ✗
    [444] Find the quotient and remainder in each of the following. $80\vert\overline{9075}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $77\vert\overline{8685}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $56\vert\o
      gold: "['-0.203125', '4.890625', '-4.0625']"
      baseline_extracted: "['-0.203125', '4.890625', '-4.0625']"  ✓
      variant_extracted:  "['0.1731', '3.612', '-3.762']"  ✗

## stats_descriptive

  n=49    baseline=21/49 (42.9%)    variant=24/49 (49.0%)    Δ=+6.1pp
  wins(v_only)=3  losses(b_only)=0  flips_total=3
  baseline len: mean=14693 p50=12509 p95=36152
  variant  len: mean=13966 p50=12383 p95=34335
  multi-box (>1 \boxed{}): baseline=33, variant=34

  Wins (3 samples — variant fixed):
    [362] Factor the difference of squares: $36x^{2}-25y^{2}=$ [ANS]
      gold: "['A', 'E', 'A', 'A', 'A']"
      variant_extracted: "['A', 'E', 'A', 'A', 'A']"
    [779] Given the function $f(x)=\begin{cases}\sin x,&\text{if } \cos x \text{ is a rational number,}\\\cos^2x,&\text{if } \cos x \text{ is an irrational number,}\end{cases}$ find the value of the integral $\
      gold: "['80', '76', '77', '78.5', '81', '78', '79.5', '82', '80.5', '83', '84.5', '80',"
      variant_extracted: "['80', '76', '77', '78.5', '81', '78', '79.5', '82', '80.5', '83', '84.5', '80',"
    [908] A typical cup of coffee contains about 100 mg of caffeine and every hour approximately 17\% is metabolized and eliminated. (a) Write $C$, the amount of caffeine in the body in mg as a function of $t$,
      gold: "['0.5', '0', '0.25', '1.25', 'Biology']"
      variant_extracted: "['0.5', '0', '0.25', '1.25', 'Biology']"

## calculus

  n=7    baseline=2/7 (28.6%)    variant=3/7 (42.9%)    Δ=+14.3pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=18873 p50=13220 p95=42394
  variant  len: mean=20558 p50=17666 p95=39297
  multi-box (>1 \boxed{}): baseline=5, variant=6

  Wins (1 samples — variant fixed):
    [498] A list of positive integers satisfies the following properties: (A) The mean of the list is $8$. (2) The median of the list is $13$. (D) The mode of the list is $15$.  Moreover, the range of the list
      gold: "['49', '44.5', '44.05', '44.005', '44']"
      variant_extracted: "['49', '44.5', '44.05', '44.005', '44']"

## prob_combi

  n=23    baseline=12/23 (52.2%)    variant=13/23 (56.5%)    Δ=+4.3pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=16824 p50=13958 p95=49100
  variant  len: mean=15691 p50=9694 p95=49853
  multi-box (>1 \boxed{}): baseline=15, variant=19

  Wins (1 samples — variant fixed):
    [884] Given $f(x)=x^2$, after performing the following transformations: shift upward 74 units and shift 59 units to the right, the new function $g(x)=$ [ANS]
      gold: "['2.75 x + 40108', '4.86 x', '4.86 x -(2.75 x + 40108)', '19009']"
      variant_extracted: "['40108+2.75x', '4.86x', '2.11x-40108', '19009']"

## number_alg

  n=6    baseline=3/6 (50.0%)    variant=4/6 (66.7%)    Δ=+16.7pp
  wins(v_only)=1  losses(b_only)=0  flips_total=1
  baseline len: mean=17094 p50=16329 p95=50277
  variant  len: mean=15414 p50=9541 p95=51610
  multi-box (>1 \boxed{}): baseline=4, variant=4

  Wins (1 samples — variant fixed):
    [103] If the distance from the town of Bree to Weathertop is 16 miles on a 45 degree upward slope, what is the elevation gain (omit units)? [ANS]
      gold: "['(-1)^(n+1)/(n^4)', '(n+2)/(n+4)']"
      variant_extracted: "['(-1)^{n+1}/n^4', '(n+2)/(n+4)']"
