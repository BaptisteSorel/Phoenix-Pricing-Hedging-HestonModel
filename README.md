# Phoenix Autocallable: Pricing & Hedging under Heston model

## Overview

This project implements a complete Pricing and Hedging engine for a Phoenix Autocallable Note under the Heston Stochastic Volatility Model.

Designed from scratch in Python, the engine simulates market paths, prices the exotic structure using Monte Carlo methods, computes sensitivities (Greeks), and performs a daily Delta-Hedging simulation to assess the PnL behavior and Gap Risk in distressed market scenarios.

**Key Deliverables:**
* `notebook.pdf`: A comprehensive report detailing the mathematical model, the structuring process, and the hedging analysis.
* `src/`: Modular Python codebase (Object-Oriented Design).

---

## The Product: Phoenix Autocallable

The Phoenix is a yield enhancement product linked to the S&P 500 (SPY). It offers a conditional high coupon in exchange for a capital risk at maturity.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Maturity** | 3 Years | Maximum life of the product. |
| **Underlying** | S&P 500 (SPY) | Equity Index Proxy. |
| **Autocall Barrier** | 100% | Early redemption if $S_t > S_0$. |
| **Coupon Barrier** | 70% | Conditional coupon payment (Memory effect). |
| **Capital Barrier** | 60% | **Put Down-and-In**: Capital at risk if $S_T < 60\%$. |
| **Coupon Rate** | 6.00% p.a. | Optimized via simulation to ensure positive bank margin. |

---

## The Model: Heston

To capture the Volatility Smile and the Skew (fat tails) inherent to equity markets, I moved beyond Black-Scholes to implement the Heston Stochastic Volatility Model.

The asset dynamics are governed by two coupled SDEs:

$$dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S$$
$$dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v$$

**Parameter Choice:**
Parameters were set to reflect a realistic equity market regime, notably with a correlation $\rho = -0.7$ to capture the leverage effect (volatility increases when the market crashes), which significantly increases the price of the "Put Down-and-In" implicit in the Phoenix.

---

## Key Features & Analysis

### 1. Structuring & Pricing
The engine uses Monte Carlo simulations ($N=10^5$) to determine the Fair Value.
* **Observation:** Initial pricing with high coupons resulted in a negative commercial margin due to the high cost of the downside protection (Digital & Put).
* **Optimization:** The coupon was adjusted to 1.5% quarterly (6% annually) to secure a robust margin (> 2%) covering operational costs and model risk.

### 2. Delta-Hedging Simulation (The "Stress Test")
The project simulates the lifecycle of the product and the PnL of the trading desk (Delta-Neutral strategy).

**Scenario: Distressed Market (Crash)**
In a scenario where the Autocall is deactivated and the market breaches the capital barrier:
* **Unhedged PnL (Orange):** Paradoxically rises as the market falls. Since the bank is *Short* the product, the drop in the product's Fair Value (liability) generates a mark-to-market profit.
* **Hedged PnL (Blue):** The Delta-Hedge not only stabilizes the PnL but generates **Alpha** via the **Gamma effect**. The portfolio monetizes volatility by dynamically rebalancing (buying low, selling high), outperforming the naked position.

*(See `notebook.pdf` for the detailed graphs and analysis)*

---

## 💻 Project Structure

The project follows a modular architecture to separate data, logic, and analytics.

```bash
Phoenix-Pricing-Heging-HestonModel/
├── main.py                  # Entry point for testing the simulation
├── notebook.ipynb           # Jupyter Notebook with full analysis & storytelling
├── notebook.pdf             # PDF export of the analysis
├── README.md                # This file
└── src/                     # Source Code
    ├── Models/
    │   ├── Heston.py        # MC Engine & Path Generation
    │   └── calibration.py   # Market data fetch & Parameter setting
    ├── Products/
    │   └── phoenix.py       # Payoff logic (OOD)
    └── Analytics/
        ├── greeks.py        # Finite Difference sensitivity computation
        └── pnl_sim.py       # Hedge Simulator
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BaptisteSorel/Phoenix-Pricing-Hedging-HestonModel.git](https://github.com/BaptisteSorel/Phoenix-Pricing-Hedging-HestonModel.git)
    cd Phoenix-Pricing-Hedging-HestonModel
    ```

2.  **Run the analysis:**
    * For the full story: Open `notebook.ipynb` in Jupyter.
    * For a quick simulation: Run the main script:
      ```bash
      python main.py
      ```
