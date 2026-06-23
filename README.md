# HKUST-FAM-CAPSTONE

## Pricing Convertible Bonds Using Trinomial Tree and Least Squares Monte Carlo

This repository contains the working materials and finalized thesis for our capstone project on convertible-bond valuation. The study develops and compares numerical methods for derivative pricing, with emphasis on instruments whose value depends on early-exercise decisions, coupon schedules, and dividend yields.

The finalized thesis is:

- `FinalizedThesis/Pricing_Convertible_Bonds_Using_Trinomial_Tree_and_Least_Squares_Monte_Carlo.pdf`

## Research focus

Convertible bonds are hybrid securities that combine fixed-income cash flows with equity conversion optionality. Their valuation is difficult because the optimal conversion policy depends on the interaction between bond redemption value, conversion value, coupon timing, and stock dividends.

This project investigates two numerical frameworks:

- **Trinomial tree methods** for backward induction and exercise-boundary extraction.
- **Least Squares Monte Carlo (LSMC)** for regression-based continuation-value estimation.

To benchmark these methods, the study first validates them on standard European, American, and Asian options, and then extends the analysis to convertible bonds under continuous and discrete coupon/dividend settings.

## Main contributions

- Validates Monte Carlo pricing on vanilla and path-dependent options.
- Introduces **importance sampling** to improve Monte Carlo efficiency for deep out-of-the-money contracts.
- Develops a **trinomial-tree framework** for convertible bonds with coupon and dividend features.
- Implements **LSMC** for convertible-bond pricing and early-conversion boundary estimation.
- Compares the methods across several coupon/dividend structures and identifies the lattice approach as the more stable benchmark in the studied settings.

## Key findings

- Monte Carlo estimates converge as the number of paths increases, but accuracy deteriorates for rare-event payoffs without variance reduction.
- Importance sampling substantially reduces error for extreme strikes.
- Trinomial trees produce smooth and economically consistent early-exercise boundaries, especially when discrete cash-flow timing matters.
- LSMC remains flexible and effective, but its regression-based continuation estimates can become less stable under frequent coupon structures.
- For the convertible-bond cases studied here, the trinomial tree is generally the stronger benchmark.

## Repository structure

- `FinalizedThesis/` — final thesis PDF.
- `FinalizedCode/LSMCandTrinomialTree/` — finalized Python implementations.
- `Week3/` to `Week10/` — weekly development notebooks, scripts, and interim reports.
- `LICENSE` — repository license.

## Finalized code

The main finalized implementations are stored in:

- `FinalizedCode/LSMCandTrinomialTree/math4996_discretedividenddiscretecoupon_final.py`
- `FinalizedCode/LSMCandTrinomialTree/math4999_lsmc_cont_everything_final.py`
- `FinalizedCode/LSMCandTrinomialTree/math4999_continuousdividenddiscretecoupon_final.py`

These files correspond to the final Monte Carlo, LSMC, and trinomial-tree experiments used in the thesis.
The mixed `math4996` / `math4999` prefixes reflect the original course labels of the source notebooks and are preserved in the finalized filenames.

## Reading guide

For a rigorous overview, start with the thesis PDF and then consult the finalized code for implementation details and parameter choices. The weekly folders preserve the development history and intermediate experiments that led to the final model.

## Acknowledgment

This project was completed as part of HKUST FAM capstone work by Wong King Long, Mishra Suman Mayank, and Lau Kam Yung.
