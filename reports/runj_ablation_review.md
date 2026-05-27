# Run J ablation — deep review

For each branch: paired baseline (Run F) vs variant (Run J branch enabled).
Wins = variant fixed; Losses = variant broke. Multi-box >1 is a regression flag.

## olympiad

  n=19    baseline=2/19 (10.5%)    variant=1/19 (5.3%)    Δ=-5.3pp
  wins(v_only)=0  losses(b_only)=1  flips_total=1
  baseline len: mean=23967 p50=22983 p95=64114
  variant  len: mean=22478 p50=16502 p95=62655
  multi-box (>1 \boxed{}): baseline=10, variant=7

  Losses (1 samples — variant broke):
    [967] A biologist has been observing a tree’s height. $13$ months into the observation, the tree was $18.95$ feet tall. $19$ months into the observation, the tree was $20.45$ feet tall. Let $x$ be the numbe
      gold: "['y = 0.25*x+15.7', '22.2', '53']"
      baseline_extracted: "['0.25x+15.7', '22.2', '53']"  ✓
      variant_extracted:  "['y=0.25x+15.7', '22.2', '53']"  ✗

## trig

  n=40    baseline=21/40 (52.5%)    variant=19/40 (47.5%)    Δ=-5.0pp
  wins(v_only)=0  losses(b_only)=2  flips_total=2
  baseline len: mean=8907 p50=6692 p95=22569
  variant  len: mean=8007 p50=6550 p95=22018
  multi-box (>1 \boxed{}): baseline=25, variant=20

  Losses (2 samples — variant broke):
    [450] There is an equiangular hexagon $OUTDIA$ such that $OU=DI$ , $UT=IA$ , $TD=AO$ are all integer side lengths. The incircles of triangles $OUT$ and $OTD$ are tangent to each other. Given that $OU,UT,TD$
      gold: "['h(x)', 'f(x)', '1/2*3^x', 'g(x)', '-2*x^3']"
      baseline_extracted: "['h', 'f', '(1)/(2)\\\\cdot3^x', 'g', '-2x^3']"  ✓
      variant_extracted:  "['h', 'f', '0.5\\\\cdot3^x', 'g', '-2x^3', 'h,f,0.5\\\\cdot3^x,g,-2x^3']"  ✗
    [788] Let $f(x) =\frac{2}{x^2 -1}$ . Find the largest positive integer $n$ such that $$$f(2) + f(3) + ... + f(n) \ge \frac{2006}{1337} - \frac{101}{n} + \frac{99}{n + 1}.$$$
      gold: "['C', 'D', 'E', 'A', 'B']"
      baseline_extracted: "['C', 'D', 'E', 'A', 'B']"  ✓
      variant_extracted:  "['4', '5', '1', '2', '3']"  ✗

## geometry

  n=50    baseline=26/50 (52.0%)    variant=18/50 (36.0%)    Δ=-16.0pp
  wins(v_only)=0  losses(b_only)=8  flips_total=8
  baseline len: mean=13159 p50=7057 p95=48045
  variant  len: mean=12016 p50=7455 p95=35229
  multi-box (>1 \boxed{}): baseline=30, variant=20

  Losses (3 samples — variant broke):
    [379] A company produce snowboards. Fixed costs are \$1320 and the cost per snowboard is \$260. An order has been placed for 6 snowboards. What should the retail price be in order for the company to break e
      gold: "['7.95110055275369']"
      baseline_extracted: '\\sqrt{63.22}'  ✓
      variant_extracted:  '7.95'  ✗
    [445] $P$ is the probability that if you flip a fair coin, $20$ heads will occur before $19$ tails. If $P=\frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers, find the remainder when $m+n$
      gold: "['8.90117918517108', '22.6980069221863']"
      baseline_extracted: "['17\\\\pi/6', '289\\\\pi/40']"  ✓
      variant_extracted:  "['8.9', '22.7', '8.90,22.70']"  ✗
    [483] Let $x,y$ be two non-negative real numbers such that $y\sqrt{2016-x^2} + x\sqrt{2016-y^2} = 2016$ . Then, the maximum possible value of $x+3y$ can be expressed as $m\sqrt{n}$ , where $n$ is not divisi
      gold: "['4.10841829841869']"
      baseline_extracted: '(555)/(43\\pi)'  ✓
      variant_extracted:  '8'  ✗

## stats_hyp_test

  n=50    baseline=16/50 (32.0%)    variant=16/50 (32.0%)    Δ=+0.0pp
  wins(v_only)=1  losses(b_only)=1  flips_total=2
  baseline len: mean=19391 p50=18184 p95=37227
  variant  len: mean=19423 p50=19015 p95=35453
  multi-box (>1 \boxed{}): baseline=43, variant=43

  Wins (1 samples — variant fixed):
    [691] How many rounds of golf do those physicians who play golf play per year? A survey of 12 physicians revealed the following numbers: 8, \quad 42, \quad 16, \quad 3, \quad 32, \quad 37, \quad 20, \quad 1
      gold: "['A', 'C']"
      variant_extracted: "['A', 'C']"

  Losses (1 samples — variant broke):
    [300] The freshman class at a major university contains $5655$ students. This was a $13 \%$ decrease from the size of the freshman class the year before. What was the size of last year's freshman class? Ans
      gold: "['2.43', '0.105', 'B', '5.76', '0.008', 'B']"
      baseline_extracted: "['2.43', '0.105', 'B', '5.76', '0.008', 'B']"  ✓
      variant_extracted:  "['2.43', '0.105', 'No', '5.76', '0.008', 'Yes']"  ✗

## stats_regression

  n=27    baseline=12/27 (44.4%)    variant=11/27 (40.7%)    Δ=-3.7pp
  wins(v_only)=0  losses(b_only)=1  flips_total=1
  baseline len: mean=16364 p50=13936 p95=38676
  variant  len: mean=15708 p50=13923 p95=40528
  multi-box (>1 \boxed{}): baseline=19, variant=17

  Losses (1 samples — variant broke):
    [444] Find the quotient and remainder in each of the following. $80\vert\overline{9075}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $77\vert\overline{8685}\ \ \ \ \ $ quotient=[ANS] remainder=[ANS] $56\vert\o
      gold: "['-0.203125', '4.890625', '-4.0625']"
      baseline_extracted: "['-0.203125', '4.890625', '-4.0625']"  ✓
      variant_extracted:  "['0.17307692307692307', '3.6115384615384616', '-3.7615384615384615']"  ✗

## stats_descriptive

  n=49    baseline=21/49 (42.9%)    variant=22/49 (44.9%)    Δ=+2.0pp
  wins(v_only)=2  losses(b_only)=1  flips_total=3
  baseline len: mean=14693 p50=12509 p95=36152
  variant  len: mean=15118 p50=13923 p95=39791
  multi-box (>1 \boxed{}): baseline=33, variant=31

  Wins (2 samples — variant fixed):
    [362] Factor the difference of squares: $36x^{2}-25y^{2}=$ [ANS]
      gold: "['A', 'E', 'A', 'A', 'A']"
      variant_extracted: "['A', 'E', 'A', 'A', 'A']"
    [905] For each paired data set, construct a scatterplot and identify the mathematical model that best fits the given data. $\begin{array}{c|ccccccc} x & 1 & 2 & 3 & 4 & 5 & 6 & 7\cr \hline y & 1 & 2.83 & 5.
      gold: "['531']"
      variant_extracted: '531'

  Losses (1 samples — variant broke):
    [524] Find the Dini derivative of $$ f(x)=\begin{cases}ax\sin^2\frac{1}{x}+bx\cos^2\frac{1}{x},&x>0,\\0,&x=0,\quad(a<b,a^{\prime}<b^{\prime}).\\\\a^{\prime}x\sin^2\frac{1}{x}+b^{\prime}x\cos^2\frac{1}{x},&x
      gold: "['A', 'A']"
      baseline_extracted: "['A', 'A']"  ✓
      variant_extracted:  "['A']"  ✗

## calculus

  n=7    baseline=2/7 (28.6%)    variant=3/7 (42.9%)    Δ=+14.3pp
  wins(v_only)=2  losses(b_only)=1  flips_total=3
  baseline len: mean=18873 p50=13220 p95=42394
  variant  len: mean=15979 p50=13698 p95=39885
  multi-box (>1 \boxed{}): baseline=5, variant=6

  Wins (2 samples — variant fixed):
    [240] The table below gives the height $h=f(t)$ in feet of a weight on a spring where $t$ is time in seconds. $\begin{array}{ccccccccccccccccc}\hline t(sec) & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11
      gold: "['T', 'F', 'F', 'T']"
      variant_extracted: "['True', 'False', 'False', 'True']"
    [498] A list of positive integers satisfies the following properties: (A) The mean of the list is $8$. (2) The median of the list is $13$. (D) The mode of the list is $15$.  Moreover, the range of the list
      gold: "['49', '44.5', '44.05', '44.005', '44']"
      variant_extracted: "['49', '44.5', '44.05', '44.005', '44']"

  Losses (1 samples — variant broke):
    [479] Construct both a $90$ \% and a $95$ \% confidence interval for $\beta_1$. $\hat{\beta}_1=42, \ s=7.3, \ SS_{xx}=59, \ n=21$ $90$ \%: [ANS] $\leq \beta_1 \leq$ [ANS] $95$ \%: [ANS] $\leq \beta_1 \leq$
      gold: "['-0.3', 'decreased', '0.3']"
      baseline_extracted: "['-0.3', '-0.3,decreased,0.3']"  ✓
      variant_extracted:  "['-0.3', 'decreased,0.3']"  ✗

## prob_combi

  n=23    baseline=12/23 (52.2%)    variant=10/23 (43.5%)    Δ=-8.7pp
  wins(v_only)=0  losses(b_only)=2  flips_total=2
  baseline len: mean=16824 p50=13958 p95=49100
  variant  len: mean=14177 p50=7952 p95=46541
  multi-box (>1 \boxed{}): baseline=15, variant=16

  Losses (2 samples — variant broke):
    [400] Let the random variables ( X_1 ,X_2 ,X_3 ,X_4 ) be independent and identically distributed, with ( P{X_i=0}=0.6 ) and ( P{X_i =1}=0.4 ), ( i=1,2,3,4 ). Let the determinant ( X=left|{{begin{array}{*{20
      gold: "['5', '3', '2', '0', '1', '4', '5', '1', '4', '0', '3', '2']"
      baseline_extracted: "['5', '3', '2', '0', '1', '4', '5', '1', '4', '0', '3', '2']"  ✓
      variant_extracted:  "['5,3,2,0,1,4', '5,1,4,0,3,2']"  ✗
    [528] A city had a population of 7,774 at the begining of 1968 and has been growing at 7.1\% per year since then. (a) Find the size of the city at the beginning of 1993. Answer: [ANS] (b) During what year w
      gold: "['B', 'D']"
      baseline_extracted: "['B', 'D']"  ✓
      variant_extracted:  "['B', 'D']"  ✗

## number_alg

  n=6    baseline=3/6 (50.0%)    variant=2/6 (33.3%)    Δ=-16.7pp
  wins(v_only)=0  losses(b_only)=1  flips_total=1
  baseline len: mean=17094 p50=16329 p95=50277
  variant  len: mean=12498 p50=6826 p95=47193
  multi-box (>1 \boxed{}): baseline=4, variant=1

  Losses (1 samples — variant broke):
    [771] Find the exact value as fraction (not a decimal approximation). $ \sec\left(\frac{-\pi}{3} \right)$=[ANS]
      gold: "['2', '2', '1', '1', '2', '2', '0', '1', '3', '1', '0', '1']"
      baseline_extracted: "['2,2,1', '1,2,2,0,1', '3,1,0,1', '2,2,1,1,2,2,0,1,3,1,0,1']"  ✓
      variant_extracted:  "['2', '2', '1', '1', '2', '2', '0', '1', '3', '1', '0', '1', '2', '2', '1', '1',"  ✗
