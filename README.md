# Mapping the Code to the PDF Case Study

How each variable in `app.py` maps directly to the case study and explain the intuition behind the modeling choices.

## Direct Mappings from Exhibit A

### 1. **Base Cost Components** (from PDF Exhibit A table)

| Variable | PDF Values | Code Implementation | Intuition |
|----------|-----------|---------------------|-----------|
| **Raw Material** | US: $40, MX: $35, CN: $30 | `mean` matches exactly, `std` = ~10% | Commodity price volatility |
| **Labor** | US: $12, MX: $8, CN: $4 | `mean` matches exactly, `std` = ~5% | Wages are relatively stable |
| **Indirect Costs** | US: $10, MX: $8, CN: $4 | `mean` matches exactly, `std` = ~5% | Overhead costs are predictable |
| **Electricity** | US: $4, MX: $3, CN: $4 | `mean` matches exactly, `std` = ~10% | Energy price fluctuations |
| **Depreciation** | US: $5, MX: $1, CN: $5 | `mean` matches exactly, `std` = ~5% | "MX lower due to existing equipment" |

### 2. **Production Yield** - Beta Distribution

**PDF Quote:** *"Production yield (till ramp is achieved) - 6 months: 80% / 90% / 95%"* and *"Production yield varies across different production sites and is a major sticking point."*

```python
'yield_params': {
    'US': {'a': 79, 'b': 20},     # → mean ≈ 0.80 (80%)
    'Mexico': {'a': 12, 'b': 1},  # → mean ≈ 0.90 (90%)  
    'China': {'a': 49, 'b': 3}    # → mean ≈ 0.95 (95%)
}
```

**Intuition:** Beta distribution is the standard for modeling percentages/yields between 0-1. The parameters `a` and `b` shape the distribution:
- Higher `a` relative to `b` = higher yield
- Larger total (a+b) = tighter distribution (more confidence)
- Mexico has wider spread (a+b=13) reflecting "questionable skill levels"

---

## Risk Factors from Case Study Text

### 3. **Logistics Costs - LOGNORMAL for China**

**PDF Quote:** *"Fluctuating transportation costs, crisis in transportation (ex. Red Sea crisis), transportation times... Shipping containers from China... costs skyrocketing as high as 3X the normal prices which is very unpredictable."*

```python
'logistics': {
    'US': {'mean': 9, 'std': 0},          # Domestic, stable
    'Mexico': {'mean': 7, 'std': 0.056},  # Minimal variation
    'China': {'dist': 'lognormal', 'mean': 12, 'std': 8}  # EXTREME volatility
}
```

**Intuition:** 
- **Lognormal distribution** captures right-tail risk (can't go below $0, but can spike 3x+)
- Normal distribution would allow negative costs (impossible)
- High `std=8` relative to `mean=12` creates fat right tail for "3X cost spikes"

---

### 4. **Currency Fluctuation Risk**

**PDF Quote:** *"Currency exchange rates and tariffs/duties... is always subject to change with evolving government policies."*

```python
'currency_std': {
    'US': 0,      # No FX risk (domestic)
    'Mexico': 0.08,  # ±8% peso volatility
    'China': 0.03    # ±3% yuan volatility (more controlled)
}
```

**Intuition:** Multiplies total cost by `(1 + normal(0, currency_std))` to simulate exchange rate fluctuations

---

### 5. **Tariff Escalation Risk**

**PDF Quote:** *"President Donald J. Trump signed a proclamation... to impose a 25% tariff"* and *"tariffs/duties... always subject to change"*

```python
'tariff': {
    'US': {'fixed': 0},
    'Mexico': {'fixed': 15.5},   # From Exhibit A
    'China': {'fixed': 15}       # From Exhibit A
},
'tariff_escal': {'mean': 0, 'std': 2}  # Future tariff uncertainty
```

**Intuition:** Base tariff from PDF + random normal variation to model policy risk

---

## Discrete Risk Events (Binary Outcomes)

### 6. **Border Crossing Delays (Mexico only)**

**PDF Quote:** *"Border crossing between MX & US was typically less than 48 hours but times seem to be increasing due to trade conflicts even within the NA region."*

```python
'border_mean': 0.83,      # Average 0.83 hours
'border_std': 0.67,       # High variability  
'border_threshold': 2,    # 2 hours is "free"
'border_cost_per_hr': 10  # $10/hr penalty after 2 hrs
```

**Intuition:** 
- Samples crossing time from normal distribution
- If time > 2 hours → adds penalty cost
- Captures "increasing times due to trade conflicts"

---

### 7. **Shipping Cancellation Risk (China only)**

**PDF Quote:** *"Shipping containers from China have increasingly become harder with cancellation at last minutes due to trade barriers"*

```python
'cancellation_prob': 0.3,      # 30% chance!
'cancellation_impact': 50      # $50 expedite cost
```

**Code comment:** *"Updated from recent shipping data (30% cancellations)"*

**Intuition:** Binary event (happens or doesn't) modeled with binomial distribution

---

### 8. **Damage Risk**

**PDF Quote:** *"As a cosmetic/appearance part on the vehicle, transportation of the part from one location to another is always subject to risk of damages and unforeseen quality issues."*

```python
'damage_prob': {
    'US': 0.01,      # 1% (shortest distance)
    'Mexico': 0.015, # 1.5%
    'China': 0.02    # 2% (longest distance)
}
```

**Intuition:** Longer shipping = higher damage probability. Cosmetic part = expensive to replace.

---

### 9. **Skills Gap (Mexico only)**

**PDF Quote:** *"The MX facilities were equally modern with lower labor costs but the overall skill levels within the region were questionable."*

```python
'skills_mean': 0,
'skills_std': 0.05  # ±5% cost variation
```

**Intuition:** Multiplies cost by `(1 + normal(0, 0.05))` to capture quality/rework cost uncertainty from skill gaps

---

### 10. **Supply Chain Disruptions**

**PDF Context:** General geopolitical tensions, Red Sea crisis, trade conflicts

```python
'disruption_prob': {
    'US': 0.05,     # 5% (most stable)
    'Mexico': 0.1,  # 10%  
    'China': 0.2    # 20% (highest geopolitical risk)
}
```

---

## Key Modeling Decisions

### Why Monte Carlo Simulation?

The PDF states: *"There is no right/wrong answer... we are looking for detailed analysis"*

**Instead of calculating:**
```
Total Cost = Raw + Labor + Indirect + ... + Tariff
```

**They model UNCERTAINTY:**
```python
# Run 10,000 scenarios
for i in range(10000):
    raw = random.normal(40, 4)  # Different each time
    labor = random.normal(12, 0.6)
    # ... + all the risk events
    total_cost[i] = ...

# Result: Distribution of possible outcomes
mean_cost = np.mean(total_cost)
worst_case = np.percentile(total_cost, 95)
```

### Distribution Choice Logic

| Distribution | Use Case | Why? |
|-------------|----------|------|
| **Normal** | Labor, raw materials, electricity | Symmetric variation around mean |
| **Lognormal** | China logistics | Can't be negative, has extreme right tail (3x spikes) |
| **Beta** | Yields (0-100%) | Bounded between 0 and 1 |
| **Binomial** | Cancellations, disruptions, damage | Yes/no events |

---

## Working Capital (Not in PDF Table)

```python
'working_capital': {
    'US': 5,
    'Mexico': 6,
    'China': 10  # Highest due to long lead times
}
```

**Intuition:** While not in Exhibit A, the "inventory carrying costs" from the PDF get split between logistics and working capital. China's longer shipping times require more inventory investment.

---

## The Bottom Line

The team took the **deterministic values from Exhibit A** and added:
1. **Uncertainty** (standard deviations based on cost category volatility)
2. **Risk events** (from the challenges section of the PDF)
3. **Appropriate probability distributions** (normal, lognormal, beta, binomial)

This transforms a simple spreadsheet comparison into a **risk-adjusted total cost of ownership analysis** that shows not just average costs but the full range of possible outcomes including worst-case scenarios.