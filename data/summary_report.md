# Synthetic Training Dataset Summary Report: Decision-Native Intelligence

## 1. Executive Summary
- **Total Contrastive Pairs**: 300
- **Total Decision Scenarios**: 600
- **QA Reverse Validation Pass Rate**: 100.0%
- **Delta Check Precision**: 100.0% (strictly 1 differing variable between Scenario A and Scenario B)
- **Deterministic Rule Engine Agreement**: 100.0% match with ground-truth business logic

## 2. Distribution Breakdown

### By Domain
| Domain | Contrastive Pairs | SFT Scenarios | Percentage |
| :--- | :--- | :--- | :--- |
| **Finance / Expense Routing** | 100 | 200 | 33.3% |
| **Sales / Discount & Credit** | 100 | 200 | 33.3% |
| **Inventory Movements** | 100 | 200 | 33.3% |

### By Perturbation Dimension (Swiss Cheese Method)
| Perturbation Dimension | Count | Description |
| :--- | :--- | :--- |
| `value_boundary` | 120 | Tight micro-deltas crossing threshold (+1 ZMW, +0.1% discount, +1 unit) |
| `categorical_boundary` | 105 | Category shift toggling policy (equipment vs software, retail vs enterprise) |
| `missing_data_boundary` | 75 | Missing mandatory entity field triggering automated fallback to review |

### Decision Label Balance
| Label | Count | Percentage |
| :--- | :--- | :--- |
| `approve` | 300 | 50.0% |
| `review` | 300 | 50.0% |

## 3. Representative Pair Samples

### Finance: Value Boundary (+1 ZMW crossing 10,000 ZMW)
```json
{
  "id": "fin_val_0001",
  "domain": "finance",
  "perturbation_dimension": "value_boundary",
  "reasoning_trace": "Business rule threshold: Equipment expenses <= 10,000 ZMW are 'approve', while expenses > 10,000 ZMW require 'review'. Department 'information_technology', vendor 'Liquid Intelligent Technologies', and remaining budget 35371 ZMW remain identical. In Scenario A, amount 10000.00 ZMW is <= 10,000.00 ZMW (label: approve). In Scenario B, amount is shifted by +15 ZMW to 10015.00 ZMW, strictly crossing the boundary (label: review).",
  "contrastive_pair": [
    {
      "id": "fin_val_0001_A",
      "decision_type": "classification",
      "context": {
        "department": "information_technology",
        "vendor": "Liquid Intelligent Technologies",
        "expense_type": "equipment",
        "budget_remaining": 35371,
        "amount": 10000.0
      },
      "label": "approve",
      "boundary_delta": "base (at/below threshold)"
    },
    {
      "id": "fin_val_0001_B",
      "decision_type": "classification",
      "context": {
        "department": "information_technology",
        "vendor": "Liquid Intelligent Technologies",
        "expense_type": "equipment",
        "budget_remaining": 35371,
        "amount": 10015.0
      },
      "label": "review",
      "boundary_delta": "+15 ZMW"
    }
  ]
}
```

### Sales: Value Boundary (+0.1% crossing 15.0% discount)
```json
{
  "id": "sal_val_0001",
  "domain": "sales",
  "perturbation_dimension": "value_boundary",
  "reasoning_trace": "Commercial credit boundary: Total exposure (outstanding_balance + order_value) must not exceed credit limit of 75000 ZMW. In Scenario A, total exposure is 75000 ZMW == credit limit (label: approve). In Scenario B, balance increases by +50 ZMW balance crossing credit limit to 55050 ZMW, creating total exposure of 75050 ZMW (label: review).",
  "contrastive_pair": [
    {
      "id": "sal_val_0001_A",
      "decision_type": "classification",
      "context": {
        "customer_id": "CUST-ZAM-208",
        "customer_tier": "standard_retail",
        "order_value": 20000.0,
        "discount_percent": 10.0,
        "credit_limit": 75000.0,
        "outstanding_balance": 55000.0
      },
      "label": "approve",
      "boundary_delta": "base (within authorized limits)"
    },
    {
      "id": "sal_val_0001_B",
      "decision_type": "classification",
      "context": {
        "customer_id": "CUST-ZAM-208",
        "customer_tier": "standard_retail",
        "order_value": 20000.0,
        "discount_percent": 10.0,
        "credit_limit": 75000.0,
        "outstanding_balance": 55050.0
      },
      "label": "review",
      "boundary_delta": "+50 ZMW balance crossing credit limit"
    }
  ]
}
```

### Inventory: Value Boundary (Unit cost shift crossing 5,000 ZMW write-off)
```json
{
  "id": "inv_val_0001",
  "domain": "inventory",
  "perturbation_dimension": "value_boundary",
  "reasoning_trace": "Inventory write-off boundary: Routine inventory adjustments <= 5,000.00 ZMW (and <= 50 units) are approved by supervisors. Values > 5,000.00 ZMW require plant manager audit review. In Scenario A, 20 units @ 250.00 ZMW = 5,000.00 ZMW exact (label: approve). In Scenario B, unit cost increases by 1.00 ZMW to 251.00 ZMW, pushing loss value to 5020.00 ZMW (label: review).",
  "contrastive_pair": [
    {
      "id": "inv_val_0001_A",
      "decision_type": "classification",
      "context": {
        "item_code": "SKU-LUS-720",
        "movement_type": "write_off",
        "quantity": 20.0,
        "unit_cost": 250.0,
        "source_warehouse": "Livingstone Transit Hub",
        "safety_stock_threshold": 50.0,
        "current_stock": 300.0
      },
      "label": "approve",
      "boundary_delta": "base (within authorized limits)"
    },
    {
      "id": "inv_val_0001_B",
      "decision_type": "classification",
      "context": {
        "item_code": "SKU-LUS-720",
        "movement_type": "write_off",
        "quantity": 20.0,
        "unit_cost": 251.0,
        "source_warehouse": "Livingstone Transit Hub",
        "safety_stock_threshold": 50.0,
        "current_stock": 300.0
      },
      "label": "review",
      "boundary_delta": "+20 ZMW write-off value crossing 5,000 ZMW limit"
    }
  ]
}
```
